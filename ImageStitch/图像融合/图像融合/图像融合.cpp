#include "opencv2/core/core.hpp"
#include "highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include<opencv2/imgcodecs.hpp>
#include<opencv2/features2d.hpp>
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#include <iostream>
#include <fstream> 
#include <string>
#include <iomanip> 
using namespace cv;
using namespace std;
using namespace detail;

int main(int argc, char** argv)
{
	double t1 = clock();
	cout << "t1 = " << t1 << endl;
	int num_images = 2;
	vector<Mat> imgs;    //����ͼ��
	Mat img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\-22.bmp");
	imgs.push_back(img);
	img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\-11.bmp");
	imgs.push_back(img);

	Ptr<FeaturesFinder> finder;    //��������Ѱ����
	//finder = new SurfFeaturesFinder();    //Ӧ��SURF����Ѱ������
	finder = new  OrbFeaturesFinder();    //Ӧ��ORB����Ѱ������
	vector<ImageFeatures> features(num_images);    //��ʾͼ������
	for (int i = 0; i < num_images; i++)
		(*finder)(imgs[i], features[i]);    //�������
	vector<MatchesInfo> pairwise_matches;    //��ʾ����ƥ����Ϣ����
	BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);    //��������ƥ������2NN����
	matcher(features, pairwise_matches);    //��������ƥ��
	double t2 = clock();
	cout << "t2=" << t2 << endl;
	HomographyBasedEstimator estimator;    //�������������
	vector<CameraParams> cameras;    //��ʾ�������
	estimator(features, pairwise_matches, cameras);    //���������������

	for (size_t i = 0; i < cameras.size(); ++i)    //ת�������ת��������������
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}
	double t3 = clock();
	cout << "t3=" << t3 << endl;
	Ptr<detail::BundleAdjusterBase> adjuster;    //����ƽ�����ȷ�������
	//adjuster = new detail::BundleAdjusterReproj();    //��ӳ������
	adjuster = new detail::BundleAdjusterRay();    //���߷�ɢ����

	adjuster->setConfThresh(1);    //����ƥ�����Ŷȣ���ֵ��Ϊ1
	(*adjuster)(features, pairwise_matches, cameras);    //��ȷ�����������

	//������ù���ƽ���Ч���ܲ�
	//��ȷ���������ԭʼ��������Ǻܴ��
	double t4 = clock();
	cout << "t4=" << t4 << endl;
	//vector<Mat> rmats;
	//for (size_t i = 0; i < cameras.size(); ++i)    //�����������ת����
		//rmats.push_back(cameras[i].R.clone());
	//waveCorrect(rmats, WAVE_CORRECT_HORIZ);    //���в���У��
	//for (size_t i = 0; i < cameras.size(); ++i)    //���������ֵ
	//	cameras[i].R = rmats[i];
	//rmats.clear();    //�����

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
	//warper_creator = new cv::PlaneWarper();    //ƽ��ͶӰ
	warper_creator = new cv::CylindricalWarper();    //����ͶӰ
	//warper_creator = new cv::SphericalWarper();    //����ͶӰ
	//warper_creator = new cv::FisheyeWarper();    //����ͶӰ
	//warper_creator = new cv::StereographicWarper();    //������ͶӰ
	double t5 = clock();
	cout << "t5=" << t5 << endl;
	//����ͼ��ӳ��任��������ӳ��ĳ߶�Ϊ����Ľ��࣬��������Ľ��඼��ͬ
	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(cameras[0].focal));
	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);    //ת������ڲ�������������
		//�Ե�ǰͼ����ͶӰ�任���õ��任���ͼ���Լ���ͼ������Ͻ�����
		corners[i] = warper->warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		cout << corners[i] << endl;
		sizes[i] = images_warped[i].size();    //�õ��ߴ�
		//�õ��任���ͼ������
		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}
	double t6 = clock();
	cout << "t6=" << t6 << endl;
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
	//	Ptr<SeamFinder> seam_finder;    //����ӷ���Ѱ����
	   //seam_finder = new NoSeamFinder();    //����Ѱ�ҽӷ���
		//seam_finder = new VoronoiSeamFinder();    //��㷨
		//seam_finder��һ����
		//seam_finder = new DpSeamFinder(DpSeamFinder::COLOR);    //��̬�淶��
		//seam_finder = new DpSeamFinder(DpSeamFinder::COLOR_GRAD);
		//ͼ�
		//seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR);
		//seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR_GRAD);
	vector<UMat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)    //ͼ����������ת��
		images_warped[i].convertTo(images_warped_f[i], CV_32F);
	double t7 = getTickCount();
	//��Ҫ��һЩ����
	Mat images1, images2;
	images_warped_f[0].copyTo(images1);
	images_warped_f[1].copyTo(images2);
	cout << corners[0] << endl;
	cout << corners[1] << endl;
	Point tl1 = corners[0];
	Point tl2 = corners[1];
	//����ȫ��ͼ��Ĵ�С
	int panoBr_, panoHe_;
	panoBr_ = tl2.x - tl1.x + images_warped[1].cols;
	panoHe_ = max(tl1.y + images_warped[0].rows, tl2.y + images_warped[1].rows) - min(tl1.y, tl2.y);
	//����ƫ����,��ͼ��1��ȫ��ͼ�����Ͻ�dx = 0��
	//ƫ��������ι���ݣ�ע���Ƿ�����
	int dx1, dx2;
	dx1 = 0;
	dx2 = tl2.x - tl1.x;
	int dy, dy1, dy2;
	dy1 = 0;
	dy2 = 0;
	dy = tl2.y - tl1.y;
	if (dy > 0)
	{
		dy2 = dy;
		dy1 = 0;
	}
	if (dy < 0)
	{
		dy1 = -dy;
		dy2 = 0;
	}
	Point unionTl_, unionBr_;
	//�����ص�����Ŀ�͸�
	Point intersectTl(std::max(tl1.x, tl2.x), std::max(tl1.y, tl2.y));

	Point intersectBr(std::min(tl1.x + images1.cols, tl2.x + images2.cols),
		std::min(tl1.y + images1.rows, tl2.y + images2.rows));
	//���if������������˵��image1��image2û���ص���������˳��ú���
	cout << "intersectTl: " << intersectTl << endl;
	cout << "intersectBr: " << intersectBr << endl;
	if (intersectTl.x >= intersectBr.x || intersectTl.y >= intersectBr.y)
		return 0; // there are no conflicts
	int height, width;
	height = intersectBr.y - intersectTl.y;
	width = intersectBr.x - intersectTl.x;
	cout << "height: " << height << endl;
	cout << "width: " << width << endl;
	//����ȫ���ص�����Ŀ�͸�
	//����ȫ���ص�������ˮƽ����ֱ�����ϵ��ݶ�
	int interSectBr_ = images_warped[0].cols - dx2;
	int interSectHe_ = panoHe_;
	cout << "ͼ��0�߶�: " << images_warped[0].rows << endl;
	cout << "ͼ��0���: " << images_warped[0].cols << endl;
	cout << "ͼ��1�߶�: " << images_warped[1].rows << endl;
	cout << "ͼ��1���: " << images_warped[1].cols << endl;
	cout << "panoBr_: " << panoBr_ << endl;
	cout << "panoHe_: " << panoHe_ << endl;
	cout << "interSecBr_: " << interSectBr_ << endl;
	cout << "interSecHe_: " << interSectHe_ << endl;
	cout << "dy: " << dy << endl;
	cout << "dx1: " << dx1 << endl;
	cout << "dx2: " << dx2 << endl;
	cout << "dy1: " << dy1 << endl;
	cout << "dy2: " << dy2 << endl;
	//������������
	Mat_<float> costV;
	costV.create(interSectHe_, interSectBr_ + 2);
	costV.setTo(0);
	//ʹ��ָ�����costV
	//һ���Ǵ�������ķ�����һ���ǲ���������ķ���
	//����������
	double t11 = getTickCount();
	if (dy > 0)
	{
		for (int y = dy2; y < interSectHe_ - dy2; ++y)
		{
			float* p1 = images1.ptr<float>(y);
			float* p2 = images2.ptr<float>(y - dy2);
			float* p3 = costV.ptr<float>(y);
			for (int x = 1; x < interSectBr_ - 1; ++x)
			{
				p3[x] = ((sqr(p1[(x + dx2) * 3] - p2[x * 3]) + sqr(p1[(x + dx2) * 3 + 1] - p2[x * 3 + 1]) + sqr(p1[(x + dx2) * 3 + 2] - p2[x * 3 + 2])) +
					(sqr(p1[(x + dx2 + 1) * 3] - p2[(x - 1) * 3]) + sqr(p1[(x + dx2 + 1) * 3 + 1] - p2[(x - 1) * 3 + 1]) + sqr(p1[(x + dx2 + 1) * 3 + 2] - p2[(x - 1) * 3 + 2]))) / 2;

			}
		}
	}
	else if (dy < 0)
	{
		//cout << "a" << endl;
		for (int y = dy1; y < interSectHe_ - dy1; ++y)
		{
			float* p1 = images1.ptr<float>(y);
			float* p2 = images2.ptr<float>(y - dy1);
			float* p3 = costV.ptr<float>(y);
			for (int x = 1; x < interSectBr_ - 1; ++x)
			{
				p3[x] = ((sqr(p1[(x + dx2) * 3] - p2[x * 3]) + sqr(p1[(x + dx2) * 3 + 1] - p2[x * 3 + 1]) + sqr(p1[(x + dx2) * 3 + 2] - p2[x * 3 + 2])) +
					(sqr(p1[(x + dx2 + 1) * 3] - p2[(x - 1) * 3]) + sqr(p1[(x + dx2 + 1) * 3 + 1] - p2[(x - 1) * 3 + 1]) + sqr(p1[(x + dx2 + 1) * 3 + 2] - p2[(x - 1) * 3 + 2]))) / 2;

			}
		}
	}
	else
	{
		//û���ص��Ĳ�����0����
		int row = min(images1.rows, images2.rows);
		for (int y = 0; y < row; ++y)
		{
			float* p1 = images1.ptr<float>(y);
			float* p2 = images2.ptr<float>(y);
			float* p3 = costV.ptr<float>(y);
			for (int x = 1; x < interSectBr_ - 1; ++x)
			{
				p3[x] = ((sqr(p1[(x + dx2) * 3] - p2[x * 3]) + sqr(p1[(x + dx2) * 3 + 1] - p2[x * 3 + 1]) + sqr(p1[(x + dx2) * 3 + 2] - p2[x * 3 + 2])) +
					(sqr(p1[(x + dx2 + 1) * 3] - p2[(x - 1) * 3]) + sqr(p1[(x + dx2 + 1) * 3 + 1] - p2[(x - 1) * 3 + 1]) + sqr(p1[(x + dx2 + 1) * 3 + 2] - p2[(x - 1) * 3 + 2]))) / 2;

			}
		}
	}

	double t12 = getTickCount();
	cout << (t12 - t11) / (getTickFrequency()) << endl;
	imwrite("costV.bmp", costV);


	//Ѱ����ѷ����
	vector<Point> seam;
	//��һ���죬��ʼ��ΪPoint(142��1098)
	Point p1(interSectBr_ / 2, 0);
	seam.push_back(p1);
	while (p1.y < interSectHe_ - 1)
	{
		float* p = costV.ptr<float>(p1.y + 1);
		float a = p[p1.x - 1];
		float b = p[p1.x];
		float c = p[p1.x + 1];

		if (a == b && a == c)
		{
			p1 = Point(p1.x, p1.y + 1);
			seam.push_back(p1);
			continue;

		}
		if (a <= b && a <= c)
		{
			p1 = Point(p1.x - 1, p1.y + 1);
			seam.push_back(p1);
			continue;

		}
		if (b <= a && b <= c)
		{
			p1 = Point(p1.x, p1.y + 1);
			seam.push_back(p1);
			continue;

		}
		if (c <= a && c <= b)
		{
			p1 = Point(p1.x + 1, p1.y + 1);
			seam.push_back(p1);
			continue;
		}
	}
	//for (int i = 0; i < seam.size(); i++)
		//cout << seam[i] << endl;

	Mat mask0, mask1;
	Mat a, b;
	cvtColor(images1, a, CV_RGB2GRAY);
	cvtColor(images2, b, CV_RGB2GRAY);
	mask0.create(a.rows, a.cols, CV_32FC1);
	mask1.create(b.rows, b.cols, CV_32FC1);
	mask0.setTo(1);
	mask1.setTo(1);
	//����mask0��mask1���Դ����ҳ��ص�����
	//����mask0���ص�����
	cout << "bbbbbbbbbbb" << endl;
	double t13 = getTickCount();
	//��ͼ��1�ص����ָ���Ȩ��
	Mat aaa;
	mask0.copyTo(aaa);
	//int count = 2;
	//int left = 0, right = 0;
	//���������ص����������
	Mat mask_r1, mask_r2;
	mask_r1.create(height, width + 2, CV_32FC1);
	mask_r2.create(height, width + 2, CV_32FC1);
	if (dy > 0)
	{
		//���ñ߽�����
		for (int y = 0; y < height; ++y)
		{
			float* p1 = mask_r1.ptr<float>(y);
			float* p2 = mask_r2.ptr<float>(y);

			p1[0] = 128;
			p2[0] = 128;
			p1[width + 1] = 128;
			p2[width + 1] = 128;
		}
		//���ص�����
		for (int y = 0; y < height; ++y)
		{
			float* p1 = a.ptr<float>(y + dy2);
			float* p2 = b.ptr<float>(y);
			float* p3 = mask_r1.ptr<float>(y);
			float* p4 = mask_r2.ptr<float>(y);

			for (int x = 1; x < width + 1; ++x)
			{
				if (p1[x + dx2 - 1] >= 20 && p2[x - 1] >= 20)
				{
					p3[x] = 255;
					p4[x] = 255;
				}
				if (p1[x + dx2 - 1] >= 20 && p2[x - 1] < 20)
				{
					p3[x] = 1;
					p4[x] = 0;
				}
				if (p1[x + dx2 - 1] < 20 && p2[x - 1] >= 20)
				{
					p3[x] = 0;
					p4[x] = 1;
				}
				if (p1[x + dx2 - 1] < 20 && p2[x - 1] < 20)
				{
					p3[x] = 1;
					p4[x] = 1;
				}
			}
		}

	}
	if (dy < 0)
	{
		//���ñ߽�����
		for (int y = 0; y < height; ++y)
		{
			float* p1 = mask_r1.ptr<float>(y);
			float* p2 = mask_r2.ptr<float>(y);

			p1[0] = 128;
			p2[0] = 128;
			p1[width + 1] = 128;
			p2[width + 1] = 128;
		}
		//���ص�����
		for (int y = 0; y < height; ++y)
		{
			float* p1 = a.ptr<float>(y);
			float* p2 = b.ptr<float>(y + dy1);
			float* p3 = mask_r1.ptr<float>(y);
			float* p4 = mask_r2.ptr<float>(y);

			for (int x = 1; x < width + 1; ++x)
			{
				if (p1[x + dx2 - 1] >= 20 && p2[x - 1] >= 20)
				{
					p3[x] = 255;
					p4[x] = 255;
				}
				if (p1[x + dx2 - 1] >= 20 && p2[x - 1] < 20)
				{
					p3[x] = 1;
					p4[x] = 0;
				}
				if (p1[x + dx2 - 1] < 20 && p2[x - 1] >= 20)
				{
					p3[x] = 0;
					p4[x] = 1;
				}
				if (p1[x + dx2 - 1] < 20 && p2[x - 1] < 20)
				{
					p3[x] = 1;
					p4[x] = 1;
				}
			}
		}
	}
	if (dy == 0)
	{
		//���ñ߽�����
		for (int y = 0; y < height; ++y)
		{
			float* p1 = mask_r1.ptr<float>(y);
			float* p2 = mask_r2.ptr<float>(y);

			p1[0] = 128;
			p2[0] = 128;
			p1[width + 1] = 128;
			p2[width + 1] = 128;
		}
		//���ص�����
		for (int y = 0; y < height; ++y)
		{
			float* p1 = a.ptr<float>(y);
			float* p2 = b.ptr<float>(y);
			float* p3 = mask_r1.ptr<float>(y);
			float* p4 = mask_r2.ptr<float>(y);

			for (int x = 1; x < width + 1; ++x)
			{
				if (p1[x + dx2 - 1] >= 10 && p2[x - 1] >= 10)
				{
					p3[x] = 255;
					p4[x] = 255;
				}
				if (p1[x + dx2 - 1] >= 10 && p2[x - 1] < 10)
				{
					p3[x] = 1;
					p4[x] = 0;
				}
				if (p1[x + dx2 - 1] < 10 && p2[x - 1] >= 10)
				{
					p3[x] = 0;
					p4[x] = 1;
				}
				if (p1[x + dx2 - 1] < 10 && p2[x - 1] < 10)
				{
					p3[x] = 1;
					p4[x] = 1;
				}
			}
		}
	}
	Mat c, d;
	mask_r1.copyTo(c);
	mask_r2.copyTo(d);

	//��̬����Ȩ��
	//��mask_r2����һ�α��������η���
	int count = 3;
	int left = 0;
	int right = 0;
	int count1 = 0;
	//if (dy > 0)
	//{
	for (int y = 0; y < height; ++y)
	{
		count = 3;
		left = 0;
		right = 0;
		while (count > 0)
		{
			float* p1 = mask_r1.ptr<float>(y);
			float* p2 = mask_r2.ptr<float>(y);
			if (count == 3)
			{
				for (int x = 1; x < width + 1; ++x)
				{
					if (p2[x] == 255 && p2[x - 1] == 0 && p2[x + 1] == 1)
					{
						left = x;
						//continue;
					}
					if ((p2[x] == 255 && p2[x - 1] == 0 && p2[x + 1] == 255) || (p2[x - 1] == 128 && p2[x] == 255 && p2[x + 1] == 255 && p2[x + 2] == 255 && p2[x + 3] == 255))
					{
						left = x;
						//continue;
					}
				}
				count--;
			}
			if (count == 2)
			{
				for (int x = 1; x < width + 1; ++x)
				{
					if (p2[x - 1] == 0 && p2[x] == 255 && p2[x + 1] == 1)
					{
						right = x;
						//continue;
					}
					if (p2[x - 1] == 255 && p2[x] == 255 && (p2[x + 1] == 1 || p2[x + 1] == 128))
					{
						right = x;
						//continue;
					}
				}
				count--;
			}

			//���η���Ȩ��
			//������ѷ���߷���Ȩ��
			if (count == 1)
			{
				for (int x = 1; x < width + 1; ++x)
				{
					if (p2[x] == 255)
					{
						if (left && left == right)
						{
							p1[x] = 1;
							p2[x] = 0;
						}
						else if (x <= (seam[y + dy2 + dy1].x + 1))
						{
							p1[x] = 1 - 0.5 * (x - left) / (seam[y + dy2 + dy1].x + 1 - left);
							p2[x] = 1 - p1[x];
						}
						else if (x > (seam[y + dy2 + dy1].x + 1) && x <= right)
						{
							//cout << "a" << endl;
							p1[x] = 0.5 * (right - x) / (right - seam[y + dy2 + dy1].x - 1);
							p2[x] = 1 - p1[x];
						}
					}
				}
				count--;
			}
		}
		count1++;
		//cout << count1 <<": "<<left << ' ' << right << endl;
	}
	//�ٶ�һЩ���ⲿ�ֽ��д���
	for (int y = 0; y < height; ++y)
	{
		float*p1 = mask_r1.ptr<float>(y);
		float*p2 = mask_r2.ptr<float>(y);
		for (int x = 0; x < width + 1; ++x)
		{
			if (p1[x] == 255)
			{
				p1[x] = 1;
				p2[x] = 0;
			}
		}
	}
	//}
	
	double t14 = getTickCount();
	cout << (t14 - t13) / (getTickFrequency()) << endl;
	cout << "ddddddddddd" << endl;

	//�������
	Mat pano;
	pano.create(panoHe_, panoBr_, CV_32FC3);
	pano.setTo(0);
	if (dy > 0)
	{
		//ιͼ��1������
		for (int y = 0; y < images1.rows; ++y)
		{
			float* p1 = images1.ptr<float>(y);
			float* p2 = pano.ptr<float>(y);
			for (int x = 0; x < dx2; ++x)
			{
				p2[3 * x] = p1[3 * x];
				p2[3 * x + 1] = p1[3 * x + 1];
				p2[3 * x + 2] = p1[3 * x + 2];
			}
		}
		//ιͼ��2������
		for (int y = dy2; y < images2.rows; ++y)
		{
			float* p1 = images2.ptr<float>(y - dy2);
			float* p2 = pano.ptr<float>(y);
			for (int x = images1.cols; x < pano.cols; ++x)
			{
				p2[3 * x] = p1[3 * (x - dx2)];
				p2[3 * x + 1] = p1[3 * (x - dx2) + 1];
				p2[3 * x + 2] = p1[3 * (x - dx2) + 2];
			}
		}
		//ι�ص���������
		for (int y = 0; y < height; ++y)
		{
			float* p1 = images1.ptr<float>(y + dy2);
			float* p2 = images2.ptr<float>(y);
			float* m1 = mask_r1.ptr<float>(y);
			float* m2 = mask_r2.ptr<float>(y);
			float* p3 = pano.ptr<float>(y + dy2);
			for (int x = dx2; x < dx2 + width; ++x)
			{
				p3[3 * x] = p1[3 * x] * m1[x - dx2 + 1] + p2[3 * (x - dx2)] * m2[(x - dx2 + 1)];
				p3[3 * x + 1] = p1[3 * x + 1] * m1[x - dx2 + 1] + p2[3 * (x - dx2) + 1] * m2[(x - dx2 + 1)];
				p3[3 * x + 2] = p1[3 * x + 2] * m1[x - dx2 + 1] + p2[3 * (x - dx2) + 2] * m2[(x - dx2 + 1)];

			}
		}
	}
	if (dy < 0)
	{
		//ιͼ��1������
		for (int y = 0; y < images1.rows; ++y)
		{
			float* p1 = images1.ptr<float>(y);
			float* p2 = pano.ptr<float>(y + dy1);
			for (int x = 0; x < dx2; ++x)
			{
				p2[3 * x] = p1[3 * x];
				p2[3 * x + 1] = p1[3 * x + 1];
				p2[3 * x + 2] = p1[3 * x + 2];
			}
		}
		//ιͼ��2������
		for (int y = 0; y < images2.rows; ++y)
		{
			float* p1 = images2.ptr<float>(y);
			float* p2 = pano.ptr<float>(y);
			for (int x = images1.cols; x < pano.cols; ++x)
			{
				p2[3 * x] = p1[3 * (x - dx2)];
				p2[3 * x + 1] = p1[3 * (x - dx2) + 1];
				p2[3 * x + 2] = p1[3 * (x - dx2) + 2];
			}
		}
		//ι�ص���������
		for (int y = 0; y < height; ++y)
		{
			float* p1 = images1.ptr<float>(y);
			float* p2 = images2.ptr<float>(y + dy1);
			float* m1 = mask_r1.ptr<float>(y);
			float* m2 = mask_r2.ptr<float>(y);
			float* p3 = pano.ptr<float>(y + dy1);
			for (int x = dx2; x < dx2 + width; ++x)
			{
				p3[3 * x] = p1[3 * x] * m1[x - dx2 + 1] + p2[3 * (x - dx2)] * m2[x - dx2 + 1];
				p3[3 * x + 1] = p1[3 * x + 1] * m1[x - dx2 + 1] + p2[3 * (x - dx2) + 1] * m2[x - dx2 + 1];
				p3[3 * x + 2] = p1[3 * x + 2] * m1[x - dx2 + 1] + p2[3 * (x - dx2) + 2] * m2[x - dx2 + 1];

			}
		}
	}
	if (dy == 0)
	{
		//ιͼ��1������
		for (int y = 0; y < images1.rows; ++y)
		{
			float* p1 = images1.ptr<float>(y);
			float* p2 = pano.ptr<float>(y + dy1);
			for (int x = 0; x < dx2; ++x)
			{
				p2[3 * x] = p1[3 * x];
				p2[3 * x + 1] = p1[3 * x + 1];
				p2[3 * x + 2] = p1[3 * x + 2];
			}
		}
		//ιͼ��2������
		for (int y = 0; y < images2.rows; ++y)
		{
			float* p1 = images2.ptr<float>(y);
			float* p2 = pano.ptr<float>(y);
			for (int x = images1.cols; x < pano.cols; ++x)
			{
				p2[3 * x] = p1[3 * (x - dx2)];
				p2[3 * x + 1] = p1[3 * (x - dx2) + 1];
				p2[3 * x + 2] = p1[3 * (x - dx2) + 2];
			}
		}
		//ι�ص���������
		for (int y = 0; y < height; ++y)
		{
			float* p1 = images1.ptr<float>(y);
			float* p2 = images2.ptr<float>(y);
			float* m1 = mask_r1.ptr<float>(y);
			float* m2 = mask_r2.ptr<float>(y);
			float* p3 = pano.ptr<float>(y);
			for (int x = dx2; x < dx2 + width; ++x)
			{
				p3[3 * x] = p1[3 * x] * m1[x - dx2 + 1] + p2[3 * (x - dx2)] * m2[x - dx2 + 1];
				p3[3 * x + 1] = p1[3 * x + 1] * m1[x - dx2 + 1] + p2[3 * (x - dx2) + 1] * m2[x - dx2 + 1];
				p3[3 * x + 2] = p1[3 * x + 2] * m1[x - dx2 + 1] + p2[3 * (x - dx2) + 2] * m2[x - dx2 + 1];

			}
		}
	}
	int aa;
	aa = 0;
	cout << "endl" << endl;
	double t8 = getTickCount();
	cout << "�ں�����ʱ��: " << (t8 - t7) / getTickFrequency() << endl;
	imwrite("pano.bmp", pano);
	system("pause");
	return 0;
}

