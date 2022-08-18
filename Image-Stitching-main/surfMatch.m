function [matchpts1,matchpts2]=surfMatch(img1,img2)    
im_ch = size(img1,3);
if im_ch > 1
    gray1 = im2double(rgb2gray(img1));
    gray2 = im2double(rgb2gray(img2));
elseif im_ch == 1
    gray1 = im2double(im1);
    gray2 = im2double(im2);
end

%% SURF
metric_threshold = 200; % use surf feature
num_octaves = 3;
num_scale_levels = 4;
%  ROI = [1 1 size(I,2) size(I,1)]; % Rectangular region of interest,分别表示（x，y，width，height），x，y为区域上角点。
points1 = detectSURFFeatures(gray1, 'MetricThreshold', metric_threshold, 'NumOctaves', num_octaves, 'NumScaleLevels', num_scale_levels);
points2 = detectSURFFeatures(gray2, 'MetricThreshold', metric_threshold, 'NumOctaves', num_octaves, 'NumScaleLevels', num_scale_levels);
[features1, valid_points1] = extractFeatures(gray1, points1);
[features2, valid_points2] = extractFeatures(gray2, points2);

[indexPairs,matchmetric] = matchFeatures(features1, features2);

%% ORB
% points1 = detectORBFeatures(gray1);
% points2 = detectORBFeatures(gray2);
% [features1, valid_points1] = extractFeatures(gray1, points1, 'Method', 'ORB');
% [features2, valid_points2] = extractFeatures(gray2, points2, 'Method', 'ORB');
% 
% [indexPairs,matchmetric] = matchFeatures(features1, features2, 'Method', 'Approximate', 'MatchThreshold', 10.0, 'MaxRatio', 0.9);

%% match
matched_points1 = valid_points1(indexPairs(:, 1), :);
matched_points2 = valid_points2(indexPairs(:, 2), :);
    
matchpts1 = double(matched_points1.Location');
matchpts2 = double(matched_points2.Location');
% matchpts1 = [X1;ones(1,size(X1, 2))];
% matchpts2 = [X2;ones(1,size(X2, 2))];


%% plot feature points
% figure(1),imshow(img1);
% hold on
% for k = 1:points1.Count
%     x = points1.Location(k,1); y = points1.Location(k,2);
%     plot(x, y, '.', 'Color', 'g', 'MarkerSize', 10);
% end
% figure(2),imshow(img2);
% hold on
% for k = 1:points2.Count
%     x = points2.Location(k,1); y = points2.Location(k,2);
%     plot(x, y, '.', 'Color', 'g', 'MarkerSize', 10);
% end
% hold off;