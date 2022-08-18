#include "opencv2/core/core.hpp"
#include "highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include<opencv2/imgcodecs.hpp>
#include<opencv2/features2d.hpp>
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
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
#include "D:\\Opencv3.4.2\\opencv\\sources\\modules\\core\\include\\opencv2\\core\\opencl\\ocl_defs.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/hal/hal.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/hal/hal.hpp"
#include <algorithm>
#include "D:\\Opencv3.4.2\\opencv\\sources\\modules\\calib3d\\include\\opencv2\\calib3d\\calib3d.hpp"
using namespace cv;
using namespace std;
using namespace detail;
typedef std::set<std::pair<int, int> > MatchesSet;
class FeaturesMatcher1
{
public:
	virtual ~FeaturesMatcher1() {}
	void operator ()(const ImageFeatures &features1, const ImageFeatures &features2,
		MatchesInfo& matches_info) {
		match(features1, features2, matches_info);
	}
	void operator ()(const std::vector<ImageFeatures> &features, std::vector<MatchesInfo> &pairwise_matches,
		const cv::UMat &mask = cv::UMat());
	bool isThreadSafe() const { return is_thread_safe_; }
	virtual void collectGarbage() {}

protected:
	FeaturesMatcher1(bool is_thread_safe = false) : is_thread_safe_(is_thread_safe) {}
	virtual void match(const ImageFeatures &features1, const ImageFeatures &features2,
		MatchesInfo& matches_info) = 0;
	bool is_thread_safe_;
};
class CpuMatcher1 : public FeaturesMatcher1
{
public:
	CpuMatcher1(float match_conf) : FeaturesMatcher1(true), match_conf_(match_conf) {}
	void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info);

private:
	float match_conf_;
};
struct MatchPairsBody1 : ParallelLoopBody
{
	MatchPairsBody1(FeaturesMatcher1 &_matcher, const std::vector<ImageFeatures> &_features,
		std::vector<MatchesInfo> &_pairwise_matches, std::vector<std::pair<int, int> > &_near_pairs)
		: matcher(_matcher), features(_features),
		pairwise_matches(_pairwise_matches), near_pairs(_near_pairs) {}

	void operator ()(const Range &r) const
	{
		cv::RNG rng = cv::theRNG(); // save entry rng state
		const int num_images = static_cast<int>(features.size());
		for (int i = r.start; i < r.end; ++i)
		{
			cv::theRNG() = cv::RNG(rng.state + i); // force "stable" RNG seed for each processed pair
			int from = near_pairs[i].first;
			int to = near_pairs[i].second;
			int pair_idx = from * num_images + to;

			matcher(features[from], features[to], pairwise_matches[pair_idx]);
			pairwise_matches[pair_idx].src_img_idx = from;
			pairwise_matches[pair_idx].dst_img_idx = to;

			size_t dual_pair_idx = to * num_images + from;

			pairwise_matches[dual_pair_idx] = pairwise_matches[pair_idx];
			pairwise_matches[dual_pair_idx].src_img_idx = to;
			pairwise_matches[dual_pair_idx].dst_img_idx = from;

			if (!pairwise_matches[pair_idx].H.empty())
				pairwise_matches[dual_pair_idx].H = pairwise_matches[pair_idx].H.inv();

			for (size_t j = 0; j < pairwise_matches[dual_pair_idx].matches.size(); ++j)
				std::swap(pairwise_matches[dual_pair_idx].matches[j].queryIdx,
					pairwise_matches[dual_pair_idx].matches[j].trainIdx);
		}
	}

	FeaturesMatcher1 &matcher;
	const std::vector<ImageFeatures> &features;
	std::vector<MatchesInfo> &pairwise_matches;
	std::vector<std::pair<int, int> > &near_pairs;

private:
	//void operator =(const MatchPairsBody1&);
};
class BestOf2NearestMatcher1 : public FeaturesMatcher1
{
public:
	BestOf2NearestMatcher1(bool try_use_gpu = false, float match_conf = 0.3f, int num_matches_thresh1 = 6,
		int num_matches_thresh2 = 6);

	//void collectGarbage();

protected:
	void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo &matches_info);

	int num_matches_thresh1_;
	int num_matches_thresh2_;
	Ptr<FeaturesMatcher1> impl_;
};

void FeaturesMatcher1::operator ()(const std::vector<ImageFeatures> &features, std::vector<MatchesInfo> &pairwise_matches,
	const UMat &mask)
{
	const int num_images = static_cast<int>(features.size());
	CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.cols == num_images && mask.rows));
	Mat_<uchar> mask_(mask.getMat(ACCESS_READ));
	if (mask_.empty())
		mask_ = Mat::ones(num_images, num_images, CV_8U);
	std::vector<std::pair<int, int> > near_pairs;
	for (int i = 0; i < num_images - 1; ++i)
		for (int j = i + 1; j < num_images; ++j)
			if (features[i].keypoints.size() > 0 && features[j].keypoints.size() > 0 && mask_(i, j))
				near_pairs.push_back(std::make_pair(i, j));

	pairwise_matches.resize(num_images * num_images);
	MatchPairsBody1 body(*this, features, pairwise_matches, near_pairs);
	if (is_thread_safe_)
		parallel_for_(Range(0, static_cast<int>(near_pairs.size())), body);
	else
		body(Range(0, static_cast<int>(near_pairs.size())));

}

BestOf2NearestMatcher1::BestOf2NearestMatcher1(bool try_use_gpu, float match_conf, int num_matches_thresh1, int num_matches_thresh2)
{
	impl_ = makePtr<CpuMatcher1>(match_conf);
	is_thread_safe_ = impl_->isThreadSafe();
	num_matches_thresh1_ = num_matches_thresh1;
	num_matches_thresh2_ = num_matches_thresh2;
}

void BestOf2NearestMatcher1::match(const ImageFeatures &features1, const ImageFeatures &features2,
	MatchesInfo &matches_info)
{
	//CV_INSTRUMENT_REGION();
	(*impl_)(features1, features2, matches_info);
	// Check if it makes sense to find homography
	if (matches_info.matches.size() < static_cast<size_t>(num_matches_thresh1_))
		return;

	// Construct point-point correspondences for homography estimation
	Mat src_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
	Mat dst_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
	for (size_t i = 0; i < matches_info.matches.size(); ++i)
	{
		const DMatch& m = matches_info.matches[i];

		Point2f p = features1.keypoints[m.queryIdx].pt;
		p.x -= features1.img_size.width * 0.5f;
		p.y -= features1.img_size.height * 0.5f;
		src_points.at<Point2f>(0, static_cast<int>(i)) = p;

		p = features2.keypoints[m.trainIdx].pt;
		p.x -= features2.img_size.width * 0.5f;
		p.y -= features2.img_size.height * 0.5f;
		dst_points.at<Point2f>(0, static_cast<int>(i)) = p;
	}

	// Find pair-wise motion
	matches_info.H = findHomography(src_points, dst_points, matches_info.inliers_mask, RANSAC);
	if (matches_info.H.empty() || std::abs(determinant(matches_info.H)) < std::numeric_limits<double>::epsilon())
		return;

	// Find number of inliers
	matches_info.num_inliers = 0;
	for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i)
		if (matches_info.inliers_mask[i])
			matches_info.num_inliers++;

	// These coeffs are from paper M. Brown and D. Lowe. "Automatic Panoramic Image Stitching
	// using Invariant Features"
	matches_info.confidence = matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());

	// Set zero confidence to remove matches between too close images, as they don't provide
	// additional information anyway. The threshold was set experimentally.
	matches_info.confidence = matches_info.confidence > 3. ? 0. : matches_info.confidence;

	// Check if we should try to refine motion
	if (matches_info.num_inliers < num_matches_thresh2_)
		return;

	// Construct point-point correspondences for inliers only
	src_points.create(1, matches_info.num_inliers, CV_32FC2);
	dst_points.create(1, matches_info.num_inliers, CV_32FC2);
	int inlier_idx = 0;
	for (size_t i = 0; i < matches_info.matches.size(); ++i)
	{
		if (!matches_info.inliers_mask[i])
			continue;

		const DMatch& m = matches_info.matches[i];

		Point2f p = features1.keypoints[m.queryIdx].pt;
		p.x -= features1.img_size.width * 0.5f;
		p.y -= features1.img_size.height * 0.5f;
		src_points.at<Point2f>(0, inlier_idx) = p;

		p = features2.keypoints[m.trainIdx].pt;
		p.x -= features2.img_size.width * 0.5f;
		p.y -= features2.img_size.height * 0.5f;
		dst_points.at<Point2f>(0, inlier_idx) = p;

		inlier_idx++;
	}

	// Rerun motion estimation on inliers only
	matches_info.H = findHomography(src_points, dst_points, RANSAC);
}

void CpuMatcher1::match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info)
{

	//�������������Ϣ�����������������Ϣ���һһ��Ӧ��
	CV_Assert(features1.descriptors.type() == features2.descriptors.type());
	CV_Assert(features2.descriptors.depth() == CV_8U || features2.descriptors.depth() == CV_32F);
	matches_info.matches.clear();
	Ptr<cv::DescriptorMatcher> matcher;
	{
		Ptr<flann::IndexParams> indexParams = makePtr<flann::KDTreeIndexParams>();
		Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>();

		if (features2.descriptors.depth() == CV_8U)
		{
			indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
			searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
		}

		matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);
	}
	std::vector< std::vector<DMatch> > pair_matches;
	MatchesSet matches;

	// Find 1->2 matches
	matcher->knnMatch(features1.descriptors, features2.descriptors, pair_matches, 2);
	int count = 0;
	for (size_t i = 0; i < pair_matches.size(); ++i)
	{
		if (pair_matches[i].size() < 2)
			continue;
		const DMatch& m0 = pair_matches[i][0];
		const DMatch& m1 = pair_matches[i][1];
		//cout << "queryIdx: " << m0.queryIdx << "  trainIdx: " << m1.trainIdx << endl;// << " distance0: " << m0.distance << " distance1: " << m1.distance
			//<< "m0.distance < (1.f - match_conf_) * m1.distance: " << (m0.distance < (1.f - match_conf_) * m1.distance) << endl;

		if (m0.distance < (1.f - match_conf_) * m1.distance)
		{
			count++;
			matches_info.matches.push_back(m0);
			matches.insert(std::make_pair(m0.queryIdx, m0.trainIdx));
		}

	}
	// Find 2->1 matches
	pair_matches.clear();
	matcher->knnMatch(features2.descriptors, features1.descriptors, pair_matches, 2);
	for (size_t i = 0; i < pair_matches.size(); ++i)
	{
		if (pair_matches[i].size() < 2)
			continue;
		const DMatch& m0 = pair_matches[i][0];
		const DMatch& m1 = pair_matches[i][1];
		if (m0.distance < (1.f - match_conf_) * m1.distance)
			if (matches.find(std::make_pair(m0.trainIdx, m0.queryIdx)) == matches.end())
				matches_info.matches.push_back(DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
	}

}


int main(int argc, char** argv)
{
	int num_images = 2;
	vector<Mat> imgs;    //����ͼ��
	Mat img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\00.bmp");
	imgs.push_back(img);
	img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\11.bmp");
	imgs.push_back(img);

	Ptr<OrbFeaturesFinder> finder;    //��������Ѱ����
	finder = new  OrbFeaturesFinder(Size(3, 1), 500);    //Ӧ��ORB����Ѱ������
	vector<ImageFeatures> features(num_images);    //��ʾͼ������
	for (int i = 0; i < num_images; i++)
		(*finder)(imgs[i], features[i]);    //�������
	vector<MatchesInfo> pairwise_matches;    //��ʾ����ƥ����Ϣ����
	BestOf2NearestMatcher1 matcher(false, 0.3f, 6, 6);    //��������ƥ������2NN����
	matcher(features, pairwise_matches);    //��������ƥ��

	Mat dispimg;    //����ͼ��ϲ���һ��ͼ����ʾ
	dispimg.create(Size(imgs[0].cols + imgs[1].cols, max(imgs[1].rows, imgs[1].rows)), CV_8UC3);
	Mat imgROI = dispimg(Rect(0, 0, (int)(imgs[0].cols), (int)(imgs[0].rows)));
	resize(imgs[0], imgROI, Size((int)(imgs[0].cols), (int)(imgs[0].rows)));
	imgROI = dispimg(Rect((int)(imgs[0].cols), 0, (int)(imgs[1].cols), (int)(imgs[1].rows)));
	resize(imgs[1], imgROI, Size((int)(imgs[1].cols), (int)(imgs[1].rows)));

	Point2f p1, p2;    //�ֱ��ʾ����ͼ���ڵ�ƥ����
	int j = 0;
	for (size_t i = 0; i < pairwise_matches[1].matches.size(); ++i)    //����ƥ����
	{
		if (!pairwise_matches[1].inliers_mask[i])    //�����ڵ㣬�������һ��ѭ��
			continue;

		const DMatch& m = pairwise_matches[1].matches[i];    //�õ��ڵ��ƥ����
		p1 = features[0].keypoints[m.queryIdx].pt;
		p2 = features[1].keypoints[m.trainIdx].pt;
		p2.x += features[0].img_size.width;    //p2�ںϲ�ͼ���ϵ�����

		line(dispimg, p1, p2, Scalar(0, 0, 255), 1, CV_AA);    //��ֱ��

		//if (j++ == 203)    //�ڵ������϶࣬����ֻ��ʾ10��
			//break;
	}

	//���ն���ʾ�ڵ������͵�Ӧ����
	cout << "�ڵ�������" << endl;
	cout << setw(10) << pairwise_matches[1].matches.size() << endl << endl;

	const double* h = reinterpret_cast<const double*>(pairwise_matches[1].H.data);
	cout << "��Ӧ����" << endl;
	cout << setw(10) << (int)(h[0] + 0.5) << setw(6) << (int)(h[1] + 0.5) << setw(6) << (int)(h[2] + 0.5) << endl;
	cout << setw(10) << (int)(h[3] + 0.5) << setw(6) << (int)(h[4] + 0.5) << setw(6) << (int)(h[5] + 0.5) << endl;
	cout << setw(10) << (int)(h[6] + 0.5) << setw(6) << (int)(h[7] + 0.5) << setw(6) << (int)(h[8] + 0.5) << endl;

	//imshow("ƥ����ʾ", dispimg);    //��ʾƥ��ͼ��
	imwrite("ԭʼƥ����ʾ.bmp", dispimg);

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

	blender = Blender::createDefault(Blender::NO, false);    //���ںϷ���
	//���ںϷ���
	blender = Blender::createDefault(Blender::FEATHER, false);
	FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
	fb->setSharpness(5);    //���������
	//cout << sizes[0] << endl;
	//cout << sizes[1] << endl;
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