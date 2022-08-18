%% Setup VLFeat toolbox and add other papers' codes.
addpath('D:\000\Drops\Stitch_Methods\Image-Stitching-main\modelspecific'); 
addpath('D:\000\Drops\Stitch_Methods\Image-Stitching-main\multigs');  % for feature match and homography

addpath('D:\000\Drops\Stitch_Methods\Image-Stitching-main\texture_mapping'); % for our texture mapping

addpath('D:\000\Drops\Stitch_Methods\Image-Stitching-main\LSD_matlab'); 

addpath('D:\000\Drops\Stitch_Methods\Image-Stitching-main\LineMatching'); % for line segments detection and match

addpath('D:\000\Drops\Stitch_Methods\Image-Stitching-main\ASIFT');%for feature point matchlist

addpath('D:\000\Drops\Stitch_Methods\Image-Stitching-main\MBS');   % for saliency detection

addpath('D:\000\Drops\Stitch_Methods\Image-Stitching-main\LSM');   % for line merge to get long line
addpath('D:\opencv-3.4.16\opencv\build\x64\vc15\bin');
% Setup VLFeat toolbox
run('D:\000\Drops\Stitch_Methods\Image-Stitching-main\vlfeat-0.9.14\toolbox\vl_setup');
