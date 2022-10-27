﻿//
//  Parameter.cpp
//  UglyMan_Stitching
//
//  Created by uglyman.nothinglo on 2015/8/15.
//  Copyright (c) 2015 nothinglo. All rights reserved.
//

#include "Parameter.h"

vector<string> getImageFileFullNamesInDir(const string& dir_name) {
	DIR* dir;
	struct dirent* ent;
	vector<string> result;

	const vector<string> image_formats = {
		".bmp", ".dib",
		".jpeg", ".jpg", ".jpe", ".JPG",
		".jp2",
		".png", ".PNG"
		".pbm", ".pgm", ".ppm",
		".sr", ".ras",
		".tiff", ".tif" };

	if ((dir = opendir(dir_name.c_str())) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			string file = string(ent->d_name);
			for (int i = 0; i < image_formats.size(); ++i) {
				if (file.length() > image_formats[i].length() &&
					image_formats[i].compare(file.substr(file.length() - image_formats[i].length(),
						image_formats[i].length())) == 0) {
					result.emplace_back(file);
				}
			}
		}
		closedir(dir);
	}
	else {
		printError("F(getImageFileFullNamesInDir) could not open directory");
	}
	return result;
}

bool isFileExist(const string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

/*
根据图片文件夹目录路径,来进行:
	所需文件夹 创建;
	得到文件夹下的图像名 列表;
	初始化一些参数;
	如果运行过,就还可以得到一些参数.(一堆int,double,二维数组) TODO
*/
Parameter::Parameter(const string& _file_name) {

	file_name = _file_name;
	file_dir = "./input-42-data/" + _file_name + "/";
	result_dir = "./input-42-data/0_results/" + _file_name + "-result/";

	_mkdir("./input-42-data/0_results/");
	_mkdir(result_dir.c_str());
#ifndef DP_LOG
	debug_dir = "./input-42-data/1_debugs/" + _file_name + "-result/";
	_mkdir("./input-42-data/1_debugs/");
	_mkdir(debug_dir.c_str());
#endif

	//stitching_parse_file_name = file_dir + _file_name + "-STITCH-GRAPH.txt";
	//#用op的txt
	stitching_parse_file_name = file_dir + _file_name + TXT_NAME;

	/* 读取文件夹下面的图片,仅仅将下层图像加入 image_file_full_names 列表.*/
	image_file_full_names = getImageFileFullNamesInDir(file_dir);

	/*** configure ***/
	grid_size = GRID_SIZE;
	down_sample_image_size = DOWN_SAMPLE_IMAGE_SIZE;

	/*** by file ***/
	/*
	如果已存在-STITCH-GRAPH.txt 文件,那就读取信息;第一次运行是不存在此文件的.
	信息有:
	一堆int double,和一个二维数组.
	*/
	if (isFileExist(stitching_parse_file_name)) {
		/*将文件读取/分析,得到解析数据放入map中.*/
		const InputParser input_parser(stitching_parse_file_name);
		/*从map中读取数据*/
		global_homography_max_inliers_dist = input_parser.get<double>("*global_homography_max_inliers_dist", &GLOBAL_HOMOGRAPHY_MAX_INLIERS_DIST);
		local_homogrpahy_max_inliers_dist = input_parser.get<double>("*local_homogrpahy_max_inliers_dist", &LOCAL_HOMOGRAPHY_MAX_INLIERS_DIST);
		local_homography_min_features_count = input_parser.get<int>("*local_homography_min_features_count", &LOCAL_HOMOGRAPHY_MIN_FEATURES_COUNT);

		images_count = input_parser.get<   int>("images_count");
		center_image_index = input_parser.get<int>("center_image_index");
		center_image_rotation_angle = input_parser.get<double>("center_image_rotation_angle");

		/*** check ***/

		assert(image_file_full_names.size() == images_count);
		assert(center_image_index >= 0 && center_image_index < images_count);
		/*************/

		/*images_match_graph_manually是个二维数组.*/
		images_match_graph_manually.resize(images_count);
		for (int i = 0; i < images_count; ++i) {
			images_match_graph_manually[i].resize(images_count, false);
			vector<int> labels = input_parser.getVec<int>("matching_graph_image_edges-" + to_string(i), false);
			for (int j = 0; j < labels.size(); ++j) {
				images_match_graph_manually[i][labels[j]] = true;
			}
		}

		/*** check ***/
		/*TODO 可能是检测images_match_graph_manually 数组是否正确.*/
		queue<int> que;
		vector<bool> label(images_count, false);
		que.push(center_image_index);
		while (que.empty() == false) {
			int n = que.front();
			que.pop();
			label[n] = true;
			for (int i = 0; i < images_count; ++i) {
				if (!label[i] && (images_match_graph_manually[n][i] || images_match_graph_manually[i][n])) {
					que.push(i);
				}
			}
		}
		assert(std::all_of(label.begin(), label.end(), [](bool i) {return i; }));

		/*************/

#ifndef DP_LOG
		cout << "center_image_index = " << center_image_index << endl;
		cout << "center_image_rotation_angle = " << center_image_rotation_angle << endl;
		cout << "images_count = " << images_count << endl;
		/*cout << "images_match_graph_manually = " << endl;
		for (int i = 0; i < images_match_graph_manually.size(); ++i) {
			for (int j = 0; j < images_match_graph_manually[i].size(); ++j) {
				cout << images_match_graph_manually[i][j] << " ";
			}
			cout << endl;
		}*/
#endif
	}
}

/*返回两两图像匹配 对应结果(二维数组) 如果,0--->1 那么只有0-1有数据,1-0没有数据*/
const vector<vector<bool> >& Parameter::getImagesMatchGraph() const {
	if (images_match_graph_manually.empty()) {
		printError("F(getImagesMatchGraph) image match graph verification [2] didn't be implemented yet");
		return images_match_graph_automatically; /* TODO */
	}
	return images_match_graph_manually;
}

/*将 两两图片的 对应两个index 放入 列表中,并返回.*/
const vector<pair<int, int> >& Parameter::getImagesMatchGraphPairList() const {
	if (images_match_graph_pair_list.empty()) {
		const vector<vector<bool> >& images_match_graph = getImagesMatchGraph();
		for (int i = 0; i < images_match_graph.size(); ++i) {
			for (int j = 0; j < images_match_graph[i].size(); ++j) {
				if (images_match_graph[i][j]) {
					//如果二维数组为true,则将 两两图片的 对应两个index组成的Pair 放入 images_match_graph_pair_list 列表中.
					images_match_graph_pair_list.emplace_back(i, j);
				}
			}
		}
	}
	return images_match_graph_pair_list;
}
