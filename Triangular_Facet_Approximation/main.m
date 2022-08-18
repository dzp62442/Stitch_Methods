close all; clear all;
run('./vlfeat-0.9.14/toolbox/vl_setup');
addpath('transforms');


%%
%%{
data_path_ = 'images/APAP-railtracks/'
exp_path = 'images/APAP-railtracks/'

% find all image files in the provided data folder
data_files = dir([data_path_ '*.JPG']);

im1 = imread([data_path_ data_files(1).name]);
im2 = imread([data_path_ data_files(2).name]);

peak_thresh = 0;
edge_thresh = 500;
ransac_threshold = 0.015;
run_LFA;
%}

%%
%%{
data_path_ = 'images/REW_racetracks/';
exp_path = 'images/REW_racetracks/';

% find all image files in the provided data folder
data_files = dir([data_path_ '*.JPG']);

im1 = imread([data_path_ data_files(1).name]);
im2 = imread([data_path_ data_files(2).name]);

peak_thresh = 0;
edge_thresh = 100;
ransac_threshold = 0.08;
run_LFA;
%}

%%
%%{
data_path_ = 'images/REW_tower/';
exp_path = 'images/REW_tower/';

% find all image files in the provided data folder
data_files = dir([data_path_ '*.jpg']);

im1 = imread([data_path_ data_files(1).name]);
im2 = imread([data_path_ data_files(2).name]);

peak_thresh = 0;
edge_thresh = 500;
ransac_threshold = 0.04;
run_LFA;
%}

%%
%%{
data_path_ = 'images/LFA_grove/';
exp_path = 'images/LFA_grove/';

% find all image files in the provided data folder
data_files = dir([data_path_ '*.jpg']);

im1 = imread([data_path_ data_files(1).name]);
im2 = imread([data_path_ data_files(2).name]);

peak_thresh = 0.007;
edge_thresh = 500;
ransac_threshold = 0.01;
run_LFA_fisheye;
%}

%%
%%{
data_path_ = 'images/LFA_stairs/';
exp_path = 'images/LFA_stairs/';

% find all image files in the provided data folder
data_files = dir([data_path_ '*.jpg']);
im1 = imread([data_path_ data_files(1).name]);
im2 = imread([data_path_ data_files(2).name]);

peak_thresh = 0.007;
edge_thresh = 500;
ransac_threshold = 0.06;
run_LFA_fisheye;
%}

%%
%%{
data_path_ = 'capture_3/overlap_imgs/';
exp_path = 'capture_3/overlap_imgs/';

% find all image files in the provided data folder
data_files = dir([data_path_ '*.jpg']);
im1 = imread([data_path_ data_files(1).name]);
im2 = imread([data_path_ data_files(2).name]);

peak_thresh = 0.007;
edge_thresh = 500;
ransac_threshold = 0.06;
run_LFA;
%}