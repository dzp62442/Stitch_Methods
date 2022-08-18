imsize1 = size(im1);
imsize2 = size(im2);

% if size(im1, 1) > 720
%     scale = 720/size(im1, 1);
%     im1 = imresize(im1, scale);
%     im2 = imresize(im2, scale);
%     imsize1 = size(im1);
%     imsize2 = size(im2);
% %     if data_type ~= 2
% %         M1(1,1) = M1(1,1) * scale;
% %         M1(2,2) = M1(2,2) * scale;
% %         M1(1,3) = M1(1,3) * scale;
% %         M1(2,3) = M1(2,3) * scale;
% %         M2(1,1) = M2(1,1) * scale;
% %         M2(2,2) = M2(2,2) * scale;
% %         M2(1,3) = M2(1,3) * scale;
% %         M2(2,3) = M2(2,3) * scale;
% %     end
% end

gray1 = im2single(rgb2gray(im1));
gray2 = im2single(rgb2gray(im2));

% feature selection
% peak_thresh = 0.00; % default: the most feature points
% peak_thresh = 0;
% edge_thresh = 500;
[f1,d1] = vl_sift(gray1, 'PeakThresh', peak_thresh, 'edgethresh', edge_thresh);
[f2,d2] = vl_sift(gray2, 'PeakThresh', peak_thresh, 'edgethresh', edge_thresh);

% feature matching
[matches, scores] = vl_ubcmatch(d1,d2) ;
X1 = f1(1:2,matches(1,:)) ; X1(3,:) = 1 ;
X2 = f2(1:2,matches(2,:)) ; X2(3,:) = 1 ;

figure(1); clf;
imshow([im1,im2], 'border', 'tight');