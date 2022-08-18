#include "opencv2/core/core.hpp"
//#include "highgui.h"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
//#include "opencv2/legacy/legacy.hpp"
#include<opencv2/opencv.hpp>
#include<opencv2/opencv_modules.hpp>
//#include<opencv2/imgcodecs.hpp>
//#include<opencv2/features2d.hpp>
//#include "opencv2/stitching/detail/autocalib.hpp"
//#include "opencv2/stitching/detail/blenders.hpp"
//#include "opencv2/stitching/detail/camera.hpp"
//#include "opencv2/stitching/detail/exposure_compensate.hpp"
//#include "opencv2/stitching/detail/matchers.hpp"
//#include "opencv2/stitching/detail/motion_estimators.hpp"
//#include "opencv2/stitching/detail/seam_finders.hpp"
//#include "opencv2/stitching/detail/util.hpp"
//#include "opencv2/stitching/detail/warpers.hpp"
//#include "opencv2/stitching/warpers.hpp"

#include <iostream>
#include <fstream> 
#include <string>
#include <iomanip> 
using namespace cv;
using namespace std;
using namespace detail;

float scale = 2707.47f;
float k[9];
float rinv[9];
float r_kinv[9];
float k_rinv[9];
float t[3];
inline
void mapForward(float x, float y, float &u, float &v)
{
	float x_ = r_kinv[0] * x + r_kinv[1] * y + r_kinv[2];
	float y_ = r_kinv[3] * x + r_kinv[4] * y + r_kinv[5];
	float z_ = r_kinv[6] * x + r_kinv[7] * y + r_kinv[8];

	u = scale * atan2f(x_, z_);
	v = scale * y_ / sqrtf(x_ * x_ + z_ * z_);
}
inline
void mapBackward(float u, float v, float &x, float &y)
{
	u /= scale;
	v /= scale;

	float x_ = sinf(u);
	float y_ = v;
	float z_ = cosf(u);

	float z;
	x = k_rinv[0] * x_ + k_rinv[1] * y_ + k_rinv[2] * z_;
	y = k_rinv[3] * x_ + k_rinv[4] * y_ + k_rinv[5] * z_;
	z = k_rinv[6] * x_ + k_rinv[7] * y_ + k_rinv[8] * z_;

	if (z > 0) { x /= z; y /= z; }
	else x = y = -1;
}
void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br)
{
	float tl_uf = (std::numeric_limits<float>::max)();
	float tl_vf = (std::numeric_limits<float>::max)();
	float br_uf = -(std::numeric_limits<float>::max)();
	float br_vf = -(std::numeric_limits<float>::max)();

	float u, v;
	for (int y = 0; y < src_size.height; ++y)
	{
		for (int x = 0; x < src_size.width; ++x)
		{

			mapForward(static_cast<float>(x), static_cast<float>(y), u, v);
			tl_uf = (std::min)(tl_uf, u); tl_vf = (std::min)(tl_vf, v);
			br_uf = (std::max)(br_uf, u); br_vf = (std::max)(br_vf, v);
		}
	}

	dst_tl.x = static_cast<int>(tl_uf);
	dst_tl.y = static_cast<int>(tl_vf);
	dst_br.x = static_cast<int>(br_uf);
	dst_br.y = static_cast<int>(br_vf);

}

void setCameraParams(InputArray _K, InputArray _R, InputArray _T = Mat::zeros(3, 1, CV_32F))
{
	Mat K = _K.getMat(), R = _R.getMat(), T = _T.getMat();

	CV_Assert(K.size() == Size(3, 3) && K.type() == CV_32F);
	CV_Assert(R.size() == Size(3, 3) && R.type() == CV_32F);
	CV_Assert((T.size() == Size(1, 3) || T.size() == Size(3, 1)) && T.type() == CV_32F);

	Mat_<float> K_(K);
	k[0] = K_(0, 0); k[1] = K_(0, 1); k[2] = K_(0, 2);
	k[3] = K_(1, 0); k[4] = K_(1, 1); k[5] = K_(1, 2);
	k[6] = K_(2, 0); k[7] = K_(2, 1); k[8] = K_(2, 2);

	Mat_<float> Rinv = R.t();
	rinv[0] = Rinv(0, 0); rinv[1] = Rinv(0, 1); rinv[2] = Rinv(0, 2);
	rinv[3] = Rinv(1, 0); rinv[4] = Rinv(1, 1); rinv[5] = Rinv(1, 2);
	rinv[6] = Rinv(2, 0); rinv[7] = Rinv(2, 1); rinv[8] = Rinv(2, 2);

	Mat_<float> R_Kinv = R * K.inv();
	r_kinv[0] = R_Kinv(0, 0); r_kinv[1] = R_Kinv(0, 1); r_kinv[2] = R_Kinv(0, 2);
	r_kinv[3] = R_Kinv(1, 0); r_kinv[4] = R_Kinv(1, 1); r_kinv[5] = R_Kinv(1, 2);
	r_kinv[6] = R_Kinv(2, 0); r_kinv[7] = R_Kinv(2, 1); r_kinv[8] = R_Kinv(2, 2);

	Mat_<float> K_Rinv = K * Rinv;
	k_rinv[0] = K_Rinv(0, 0); k_rinv[1] = K_Rinv(0, 1); k_rinv[2] = K_Rinv(0, 2);
	k_rinv[3] = K_Rinv(1, 0); k_rinv[4] = K_Rinv(1, 1); k_rinv[5] = K_Rinv(1, 2);
	k_rinv[6] = K_Rinv(2, 0); k_rinv[7] = K_Rinv(2, 1); k_rinv[8] = K_Rinv(2, 2);

	Mat_<float> T_(T.reshape(0, 3));
	t[0] = T_(0, 0); t[1] = T_(1, 0); t[2] = T_(2, 0);
}

Rect buildMaps(Size src_size, InputArray K, InputArray R, OutputArray _xmap, OutputArray _ymap)
{
	setCameraParams(K, R);
	Point dst_tl, dst_br;
	detectResultRoi(src_size, dst_tl, dst_br);

	_xmap.create(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, CV_32F);
	_ymap.create(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, CV_32F);

	Mat xmap = _xmap.getMat(), ymap = _ymap.getMat();
	float x, y;
	for (int v = dst_tl.y; v <= dst_br.y; ++v)
	{
		for (int u = dst_tl.x; u <= dst_br.x; ++u)
		{
			mapBackward(static_cast<float>(u), static_cast<float>(v), x, y);
			xmap.at<float>(v - dst_tl.y, u - dst_tl.x) = x;
			ymap.at<float>(v - dst_tl.y, u - dst_tl.x) = y;
		}
	}

	return Rect(dst_tl, dst_br);
}
Point warp(InputArray src, InputArray K, InputArray R, int interp_mode, int border_mode,
	OutputArray dst)
{
	UMat xmap, ymap;
	Rect dst_roi = buildMaps(src.size(), K, R, xmap, ymap);
	dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
	//�������
	Mat a, b, c, d;
	xmap.copyTo(a);
	ymap.copyTo(b);
	imwrite("xmap.bmp", xmap);
	imwrite("ymap.bmp", ymap);
	remap(src, dst, xmap, ymap, interp_mode, border_mode);
	src.copyTo(c);
	dst.copyTo(d);
	return dst_roi.tl();
}
int main()
{
	int num_images = 2;
	vector<Mat> imgs;    //����ͼ��
	Mat img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\11.bmp");
	imgs.push_back(img);
	img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\22.bmp");
	imgs.push_back(img);

	Ptr<FeaturesFinder> finder = new  OrbFeaturesFinder();    //��������Ѱ����
	vector<ImageFeatures> features(num_images);    //��ʾͼ������
	for (int i = 0; i < num_images; i++)
		(*finder)(imgs[i], features[i]);    //�������
	vector<MatchesInfo> pairwise_matches;    //��ʾ����ƥ����Ϣ����
	BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);    //��������ƥ������2NN����
	matcher(features, pairwise_matches);    //��������ƥ��

	HomographyBasedEstimator estimator;    //�������������
	vector<CameraParams> cameras;    //��ʾ�������
	estimator(features, pairwise_matches, cameras);    //���������������

	for (size_t i = 0; i < cameras.size(); ++i)    //ת�������ת��������������
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}
	Ptr<detail::BundleAdjusterBase> adjuster;    //����ƽ�����ȷ�������
	//adjuster = new detail::BundleAdjusterReproj();    //��ӳ������
	adjuster = new detail::BundleAdjusterRay();    //���߷�ɢ����

	adjuster->setConfThresh(1);    //����ƥ�����Ŷȣ���ֵ��Ϊ1
	(*adjuster)(features, pairwise_matches, cameras);    //��ȷ�����������
	//cout << "2: " << cameras[2].R << endl;
	//cout << "3: " << cameras[3].R << endl;
	//cout << "4: " << cameras[4].R << endl;
	//cout << "5: " << cameras[5].R << endl;
	//cout << "6: " << cameras[6].R << endl;
	//������ù���ƽ���Ч���ܲ�
	//��ȷ���������ԭʼ��������Ǻܴ��



	vector<Point> corners(num_images);    //��ʾӳ��任��ͼ������Ͻ�����
	vector<UMat> masks_warped(num_images);    //��ʾӳ��任���ͼ������
	vector<UMat> images_warped(num_images);    //��ʾӳ��任���ͼ��
	vector<Size> sizes(num_images);    //��ʾӳ��任���ͼ��ߴ�
	vector<Mat> masks(num_images);    //��ʾԴͼ������

	for (int i = 0; i < num_images; ++i)    //��ʼ��Դͼ������
	{
		masks[i].create(imgs[i].size(), CV_8U);    //����ߴ��С
		masks[i].setTo(Scalar::all(255));    //ȫ����ֵΪ255����ʾԴͼ����������ʹ��
	}

	Ptr<WarperCreator> warper_creator;    //����ͼ��ӳ��任������

	warper_creator = new cv::CylindricalWarper();    //����ͶӰ

	//����ͼ��ӳ��任��������ӳ��ĳ߶�Ϊ����Ľ��࣬��������Ľ��඼��ͬ
	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(cameras[0].focal));
	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);    //ת������ڲ�������������
		//�Ե�ǰͼ����ͶӰ�任���õ��任���ͼ���Լ���ͼ������Ͻ�����
		//�����￪ʼ
		corners[i] = warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();    //�õ��ߴ�
		//�õ��任���ͼ������
		warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}
	imgs.clear();    //�����
	masks.clear();

	//�����عⲹ������Ӧ�����油������
	Ptr<ExposureCompensator> compensator =
		ExposureCompensator::createDefault(ExposureCompensator::GAIN);
	compensator->feed(corners, images_warped, masks_warped);    //�õ��عⲹ����
	for (int i = 0; i < num_images; ++i)    //Ӧ���عⲹ��������ͼ������عⲹ��
	{
		compensator->apply(i, corners[i], images_warped[i], masks_warped[i]);
	}

	//�ں��棬���ǻ���Ҫ�õ�ӳ��任ͼ������masks_warped���������Ϊ�ñ������һ������masks_seam
	vector<UMat> masks_seam(num_images);
	for (int i = 0; i < num_images; i++)
		masks_warped[i].copyTo(masks_seam[i]);
	Ptr<SeamFinder> seam_finder;    //����ӷ���Ѱ����
   //seam_finder = new NoSeamFinder();    //����Ѱ�ҽӷ���
	//seam_finder = new VoronoiSeamFinder();    //��㷨
	//seam_finder��һ����
	//seam_finder = new DpSeamFinder(DpSeamFinder::COLOR);    //��̬�淶��
	//seam_finder = new DpSeamFinder(DpSeamFinder::COLOR_GRAD);
	//ͼ�
	seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR);
	//seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR_GRAD);
	vector<UMat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)    //ͼ����������ת��
		images_warped[i].convertTo(images_warped_f[i], CV_32F);
	//�õ��ӷ��ߵ�����ͼ��masks_warped

	seam_finder->find(images_warped_f, corners, masks_warped);
	
	vector<Mat> images_warped_s(num_images);
	Ptr<Blender> blender;    //����ͼ���ں���



	//blender = Blender::createDefault(Blender::MULTI_BAND, false);    //��Ƶ���ں�
	//MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
	//mb->setNumBands(4);   //����Ƶ������������������


	blender = Blender::createDefault(Blender::NO, false);    //���ںϷ���
	//���ںϷ���
	blender = Blender::createDefault(Blender::FEATHER, false);
	FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
	fb->setSharpness(0.1);    //���������
	blender->prepare(corners, sizes);    //����ȫ��ͼ������

	//���ںϵ�ʱ������Ҫ�����ڽӷ���������д�������һ����Ѱ�ҽӷ��ߺ�õ�������ı߽���ǽӷ��ߴ���������ǻ���Ҫ�ڽӷ������࿪��һ�����������ںϴ�����һ������̶��𻯷�����Ϊ�ؼ�
	//Ӧ�������㷨��С�������
	vector<Mat> dilate_img(num_images);
	Mat element = getStructuringElement(MORPH_RECT, Size(20, 20));    //����ṹԪ��
	vector<Mat> a(num_images);
	vector<Mat> b(num_images);
	vector<Mat> c(num_images);


	for (int k = 0; k < num_images; k++)
	{
		images_warped_f[k].convertTo(images_warped_s[k], CV_16S);    //�ı���������
		dilate(masks_seam[k], masks_seam[k], element);    //��������
		//ӳ��任ͼ����������ͺ�������ࡰ�롱���Ӷ�ʹ��չ������������ڽӷ������࣬�����߽紦����Ӱ��
		masks_seam[k].copyTo(a[k]);
		masks_warped[k].copyTo(b[k]);
		//masks_seam[k] = masks_seam[k] & masks_warped[k];
		c[k] = a[k] & b[k];
		c[k].copyTo(masks_seam[k]);
		blender->feed(images_warped_s[k], masks_seam[k], corners[k]);    //��ʼ������
	}


	masks_seam.clear();    //���ڴ�
	images_warped_s.clear();
	masks_warped.clear();
	images_warped_f.clear();

	Mat result, result_mask;
	//����ںϲ������õ�ȫ��ͼ��result����������result_mask
	blender->blend(result, result_mask);

	imwrite("pano.jpg", result);    //�洢ȫ��ͼ��

	system("pause");
	return 0;
}
