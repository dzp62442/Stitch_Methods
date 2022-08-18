% load_data;

feature_sift;

comp_KRt;

cal_globalparas;

if ~exist([exp_path '\\res'], 'dir')
    mkdir([exp_path '\\res']) ;
end

% globla map
mosaic_global;

% local map
mosaic_local;