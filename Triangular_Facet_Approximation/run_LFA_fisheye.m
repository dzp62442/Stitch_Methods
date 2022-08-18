% load_data;

feature_sift;

comp_KRt_fisheye;

cal_globalparas_fisheye;

if ~exist([exp_path '\\res'], 'dir')
    mkdir([exp_path '\\res']) ;
end
% globla map
mosaic_global_fisheye;

% local map
mosaic_local_fisheye;