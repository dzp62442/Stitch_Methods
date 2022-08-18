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
//#include "opencv2/stitching/detail/motion_estimators.hpp"
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

void focalsFromHomography1(const Mat& H, double &f0, double &f1, bool &f0_ok, bool &f1_ok)
{
	CV_Assert(H.type() == CV_64F && H.size() == Size(3, 3));

	const double* h = H.ptr<double>();

	double d1, d2; // Denominators ��ĸ
	double v1, v2; // Focal squares value candidates ����ƽ��ֵ��ѡ

	f1_ok = true;
	d1 = h[6] * h[7];
	d2 = (h[7] - h[6]) * (h[7] + h[6]);
	v1 = -(h[0] * h[1] + h[3] * h[4]) / d1;
	v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2;
	if (v1 < v2) std::swap(v1, v2);
	if (v1 > 0 && v2 > 0) f1 = std::sqrt(std::abs(d1) > std::abs(d2) ? v1 : v2);
	else if (v1 > 0) f1 = std::sqrt(v1);
	else f1_ok = false;

	f0_ok = true;
	d1 = h[0] * h[3] + h[1] * h[4];
	d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4];
	v1 = -h[2] * h[5] / d1;
	v2 = (h[5] * h[5] - h[2] * h[2]) / d2;
	if (v1 < v2) std::swap(v1, v2);
	if (v1 > 0 && v2 > 0) f0 = std::sqrt(std::abs(d1) > std::abs(d2) ? v1 : v2);
	else if (v1 > 0) f0 = std::sqrt(v1);
	else f0_ok = false;
}
void estimateFocal1(const std::vector<ImageFeatures> &features, const std::vector<MatchesInfo> &pairwise_matches,
	std::vector<double> &focals)
{
	const int num_images = static_cast<int>(features.size());
	focals.resize(num_images);

	std::vector<double> all_focals;

	for (int i = 0; i < num_images; ++i)
	{
		for (int j = 0; j < num_images; ++j)
		{
			cout << "i*num_images + j: " << (i * num_images + j) << endl;
			const MatchesInfo &m = pairwise_matches[i*num_images + j];
			if (m.H.empty())
				continue;
			cout << "m.H: " << m.H << endl;
			double f0, f1;
			bool f0ok, f1ok;
			focalsFromHomography1(m.H, f0, f1, f0ok, f1ok);
			if (f0ok && f1ok)
				all_focals.push_back(std::sqrt(f0 * f1));
		}
	}
	cout << pairwise_matches[0].H << endl;
	cout << pairwise_matches[1].H << endl;
	cout << pairwise_matches[2].H << endl;
	cout << pairwise_matches[3].H << endl;

	if (static_cast<int>(all_focals.size()) >= num_images - 1)
	{
		double median;

		std::sort(all_focals.begin(), all_focals.end());
		if (all_focals.size() % 2 == 1)
			median = all_focals[all_focals.size() / 2];
		else
			median = (all_focals[all_focals.size() / 2 - 1] + all_focals[all_focals.size() / 2]) * 0.5;

		for (int i = 0; i < num_images; ++i)
			focals[i] = median;
	}
	else
	{
		double focals_sum = 0;
		for (int i = 0; i < num_images; ++i)
			focals_sum += features[i].img_size.width + features[i].img_size.height;
		for (int i = 0; i < num_images; ++i)
			focals[i] = focals_sum / num_images;
	}
}


struct IncDistance
{
	IncDistance(std::vector<int> &vdists) : dists(&vdists[0]) {}
	void operator ()(const GraphEdge &edge) { dists[edge.to] = dists[edge.from] + 1; }
	int* dists;
};
class  Estimator
{
public:
	virtual ~Estimator() {}
	bool operator ()(const std::vector<ImageFeatures> &features,
		const std::vector<MatchesInfo> &pairwise_matches,
		std::vector<CameraParams> &cameras)
	{
		return estimate(features, pairwise_matches, cameras);
	}

protected:
	virtual bool estimate(const std::vector<ImageFeatures> &features,
		const std::vector<MatchesInfo> &pairwise_matches,
		std::vector<CameraParams> &cameras) = 0;
};


class CV_EXPORTS HomographyBasedEstimator1 : public Estimator
{
public:
	HomographyBasedEstimator1(bool is_focals_estimated = false)
		: is_focals_estimated_(is_focals_estimated) {}

private:
	virtual bool estimate(const std::vector<ImageFeatures> &features,
		const std::vector<MatchesInfo> &pairwise_matches,
		std::vector<CameraParams> &cameras);

	bool is_focals_estimated_;
};
void findMaxSpanningTree(int num_images, const std::vector<MatchesInfo> &pairwise_matches,
	Graph &span_tree, std::vector<int> &centers)
{
	Graph graph(num_images);
	std::vector<GraphEdge> edges;

	// Construct images graph and remember its edges
	for (int i = 0; i < num_images; ++i)
	{
		for (int j = 0; j < num_images; ++j)
		{
			if (pairwise_matches[i * num_images + j].H.empty())
				continue;
			float conf = static_cast<float>(pairwise_matches[i * num_images + j].num_inliers);
			graph.addEdge(i, j, conf);
			edges.push_back(GraphEdge(i, j, conf));
		}
	}

	DisjointSets comps(num_images);
	span_tree.create(num_images);
	std::vector<int> span_tree_powers(num_images, 0);

	// Find maximum spanning tree
	sort(edges.begin(), edges.end(), std::greater<GraphEdge>());
	for (size_t i = 0; i < edges.size(); ++i)
	{
		int comp1 = comps.findSetByElem(edges[i].from);
		int comp2 = comps.findSetByElem(edges[i].to);
		if (comp1 != comp2)
		{
			comps.mergeSets(comp1, comp2);
			span_tree.addEdge(edges[i].from, edges[i].to, edges[i].weight);
			span_tree.addEdge(edges[i].to, edges[i].from, edges[i].weight);
			span_tree_powers[edges[i].from]++;
			span_tree_powers[edges[i].to]++;
		}
	}

	// Find spanning tree leafs
	std::vector<int> span_tree_leafs;
	for (int i = 0; i < num_images; ++i)
		if (span_tree_powers[i] == 1)
			span_tree_leafs.push_back(i);

	// Find maximum distance from each spanning tree vertex
	std::vector<int> max_dists(num_images, 0);
	std::vector<int> cur_dists;
	for (size_t i = 0; i < span_tree_leafs.size(); ++i)
	{
		cur_dists.assign(num_images, 0);
		span_tree.walkBreadthFirst(span_tree_leafs[i], IncDistance(cur_dists));
		for (int j = 0; j < num_images; ++j)
			max_dists[j] = std::max(max_dists[j], cur_dists[j]);
	}

	// Find min-max distance
	int min_max_dist = max_dists[0];
	for (int i = 1; i < num_images; ++i)
		if (min_max_dist > max_dists[i])
			min_max_dist = max_dists[i];

	// Find spanning tree centers
	centers.clear();
	for (int i = 0; i < num_images; ++i)
		if (max_dists[i] == min_max_dist)
			centers.push_back(i);
	CV_Assert(centers.size() > 0 && centers.size() <= 2);
}

struct CalcRotation
{
	CalcRotation(int _num_images, const std::vector<MatchesInfo> &_pairwise_matches, std::vector<CameraParams> &_cameras)
		: num_images(_num_images), pairwise_matches(&_pairwise_matches[0]), cameras(&_cameras[0]) {}

	void operator ()(const GraphEdge &edge)
	{
		int pair_idx = edge.from * num_images + edge.to;

		Mat_<double> K_from = Mat::eye(3, 3, CV_64F);
		K_from(0, 0) = cameras[edge.from].focal;
		K_from(1, 1) = cameras[edge.from].focal * cameras[edge.from].aspect;
		K_from(0, 2) = cameras[edge.from].ppx;
		K_from(1, 2) = cameras[edge.from].ppy;

		Mat_<double> K_to = Mat::eye(3, 3, CV_64F);
		K_to(0, 0) = cameras[edge.to].focal;
		K_to(1, 1) = cameras[edge.to].focal * cameras[edge.to].aspect;
		K_to(0, 2) = cameras[edge.to].ppx;
		K_to(1, 2) = cameras[edge.to].ppy;

		Mat R = K_from.inv() * pairwise_matches[pair_idx].H.inv() * K_to;
		cameras[edge.to].R = cameras[edge.from].R * R;
	}

	int num_images;
	const MatchesInfo* pairwise_matches;
	CameraParams* cameras;
};


bool HomographyBasedEstimator1::estimate(
	const std::vector<ImageFeatures> &features,
	const std::vector<MatchesInfo> &pairwise_matches,
	std::vector<CameraParams> &cameras)
{
	const int num_images = static_cast<int>(features.size());
	//����û�б�����
	if (!is_focals_estimated_)
	{
		// Estimate focal length and set it for all cameras
		std::vector<double> focals;
		estimateFocal1(features, pairwise_matches, focals);
		cameras.assign(num_images, CameraParams());
		for (int i = 0; i < num_images; ++i)
			cameras[i].focal = focals[i];
	}
	else
	{
		for (int i = 0; i < num_images; ++i)
		{
			cameras[i].ppx -= 0.5 * features[i].img_size.width;
			cameras[i].ppy -= 0.5 * features[i].img_size.height;
		}
	}

	// Restore global motion
	Graph span_tree;
	std::vector<int> span_tree_centers;
	findMaxSpanningTree(num_images, pairwise_matches, span_tree, span_tree_centers);
	span_tree.walkBreadthFirst(span_tree_centers[0], CalcRotation(num_images, pairwise_matches, cameras));

	// As calculations were performed under assumption that p.p. is in image center
	for (int i = 0; i < num_images; ++i)
	{
		cameras[i].ppx += 0.5 * features[i].img_size.width;
		cameras[i].ppy += 0.5 * features[i].img_size.height;
	}
	return true;
}

int main(int argc, char** argv)
{
	int num_images = 2;
	vector<Mat> imgs;    //����ͼ��
	Mat img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\11.bmp");
	imgs.push_back(img);
	img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\22.bmp");
	imgs.push_back(img);

	Ptr<FeaturesFinder> finder;    //��������Ѱ����
	//finder = new SurfFeaturesFinder();    //Ӧ��SURF����Ѱ������
	finder = new  OrbFeaturesFinder();    //Ӧ��ORB����Ѱ������
	vector<ImageFeatures> features(num_images);    //��ʾͼ������
	cout << "a" << endl;
	for (int i = 0; i < num_images; i++)
		(*finder)(imgs[i], features[i]);    //�������
	cout << "c" << endl;

	vector<MatchesInfo> pairwise_matches;    //��ʾ����ƥ����Ϣ����
	cout << "a: " << pairwise_matches.size() << endl;
	BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);    //��������ƥ������2NN����
	matcher(features, pairwise_matches);    //��������ƥ��
	cout << "b: " << pairwise_matches.size() << endl;

	double t2 = clock();
	cout << "t2=" << t2 << endl;
	//cout << "diff1=" << t2 - t1 << endl;
	HomographyBasedEstimator1 estimator;    //�������������
	vector<CameraParams> cameras(2);    //��ʾ�������
	cout << "cameras[0].R: " << cameras[0].R << endl;
	cout << "cameras[1].R: " << cameras[1].R << endl;
	cout << "cameras[0].focal: " << cameras[0].focal << endl;
	cout << "cameras[1].focal: " << cameras[1].focal << endl;
	cout << "cameras[0].t: " << cameras[0].t << endl;
	cout << "cameras[1].t: " << cameras[1].t << endl;
	estimator(features, pairwise_matches, cameras);    //���������������
	cout << "cameras[0].R: " << cameras[0].R << endl;
	cout << "cameras[1].R: " << cameras[1].R << endl;
	cout << "cameras[0].focal: " << cameras[0].focal << endl;
	cout << "cameras[1].focal: " << cameras[1].focal << endl;
	cout << "cameras[0].t: " << cameras[0].t << endl;
	cout << "cameras[1].t: " << cameras[1].t << endl;

	for (size_t i = 0; i < cameras.size(); ++i)    //ת�������ת��������������
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}
	cout << cameras.size() << endl; //7,��Ӧ����6����
	cout << cameras[0].R << endl;
	double t3 = clock();
	cout << "t3=" << t3 << endl;
	//Ptr<detail::BundleAdjusterBase> adjuster;    //����ƽ�����ȷ�������
	//adjuster = new detail::BundleAdjusterReproj();    //��ӳ������
	//adjuster = new detail::BundleAdjusterRay();    //���߷�ɢ����

	//adjuster->setConfThresh(1);    //����ƥ�����Ŷȣ���ֵ��Ϊ1
	//(*adjuster)(features, pairwise_matches, cameras);    //��ȷ�����������
	//cout << "0: " << cameras[0].R << endl;
	//cout << "1: " << cameras[1].R << endl;
	//cout << "2: " << cameras[2].R << endl;
	//cout << "3: " << cameras[3].R << endl;
	//cout << "4: " << cameras[4].R << endl;
	//cout << "5: " << cameras[5].R << endl;
	//cout << "6: " << cameras[6].R << endl;
	//������ù���ƽ���Ч���ܲ�
	//��ȷ���������ԭʼ��������Ǻܴ��

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
	//mb->setNumBands(4);   //����Ƶ������������������


	blender = Blender::createDefault(Blender::NO, false);    //���ںϷ���
	//���ںϷ���
	blender = Blender::createDefault(Blender::FEATHER, false);
	FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
	fb->setSharpness(0.1);    //���������
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
	double t7 = clock();
	cout << "t7=" << t7 << endl;
	system("pause");
	return 0;
}
