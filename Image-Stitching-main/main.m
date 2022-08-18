% -------------------------------------------------------------------------------------------
% Leveraging Line-point Consistence to Preserve Structures for Wide Parallax Image Stitching
% -------------------------------------------------------------------------------------------
% This main.m implements the image stitching method propsed in:
% Leveraging Line-point Consistence to Preserve Structures for Wide
% Parallax Image Stitching
% In Conference on Computer Vision and Pattern Recognition, 2021
% 
% The program is free for non-commercial academic use. Any commercial use
% is strictly prohibited without the authors' consent. Please acknowledge
% the authors by citing the above paper in any academic publications that
% have made use of this package or part of it.
% 
% If you encounter any problems or questions please email to 
% ZhengJun.fqgw@gmail.com or zhaohaotian0210@mail.dlut.edu.cn or tsy1685212276@gmail.com
clear; clc; close all;
addNeedPaths;
%% Parameters of energy minimization (mesh deformation)
parameters.grid_size = 25;   % grid size of mesh deformation % 网格尺寸
parameters.point_align = 1;  % point alignment term % 点对齐项系数
parameters.line_align = 50;   % line alignment term % 线对齐项系数
parameters.perspective = 50; % perspective term
parameters.projective = 100; % projective term 
parameters.locallines = 50;  % local term % 局部线保持项系数
parameters.globallines = 100;  % global term % 全局线保持项系数

parameters.line_threshold = 30;  % threshold for length of matches lines

%% Images to stitch.
tar = 1;
ref = 2;  % choose the target and reference image
pathname = strcat('D:\000\Drops\GradProj_4\capture_3\precut_imgs\');
% pathname = strcat('D:\000\Drops\Stitch_Methods\Image-Stitching-main\Imgs\ANAP-building\');
outpath = strcat(pathname, 'results\');
imgs_format = '*.JPg'; 
dir_folder = dir(strcat(pathname, imgs_format));
if ~exist(outpath,'dir')
    mkdir(outpath);
end
path1 = sprintf('%s%s',pathname,dir_folder(tar).name);
path2 = sprintf('%s%s',pathname,dir_folder(ref).name);

%% Read images.
fprintf('> Reading images...');tic;
img1 = im2double(imread(path1));
img2 = im2double(imread(path2));
fprintf('done (%fs)\n',toc);
% Resolution/grid-size for the mapping function (divide it into C1*C2 cells).
C1 = ceil(size(img1,1)/parameters.grid_size);
C2 = ceil(size(img1,2)/parameters.grid_size);

%% detect and match SURF features for line matching
% [surfpts1, surfpts2] = surfMatch(img1, img2);
% [matches_1, matches_2] = multiSample_APAP(surfpts1, surfpts2, img1, img2);

%% 读取SuperGlue特征点匹配
matches_1 = csvread('D:\000\Drops\GradProj_4\capture_3\refine_match\matches_1.csv');
matches_2 = csvread('D:\000\Drops\GradProj_4\capture_3\refine_match\matches_2.csv');
matches_1 = matches_1';  % 转置
matches_2 = matches_2';

%% detect and match line segments
% [line_match1, line_match2, lpts1, lpts2] = twoLineMatch(path1, path2, matches_1, matches_2, parameters);

%% 读取MLS特征线匹配
line_match1 = csvread('D:\000\Drops\GradProj_4\capture_3\refine_match\line_match1.csv');
line_match2 = csvread('D:\000\Drops\GradProj_4\capture_3\refine_match\line_match2.csv');

%% plot match points
figure;
imshow([img1 img2]);
title('show match points');
hold on;
for i=1:size(matches_1, 2)
    plot(matches_1(1,i),matches_1(2,i),'go','LineWidth',1);  % 画点
    plot(matches_2(1,i)+size(img1,2),matches_2(2,i),'go','LineWidth',1);  % 画点
    plot([matches_1(1,i) matches_2(1,i)+size(img1,2)],[matches_1(2,i) matches_2(2,i)],'g-');  % 画线
end

%% plot match lines
figure;
imshow([img1 img2]);
title('show match lines');
hold on;
for k=1:size(line_match1, 1)
    x1 = line_match1(k,1); y1 = line_match1(k,2);
    x2 = line_match1(k,3); y2 = line_match1(k,4);
    plot([x1,x2], [y1,y2], 'LineWidth', 2, 'Color', [1, 0, 0]);
    x1 = line_match2(k,1)+size(img1,2); y1 = line_match2(k,2);
    x2 = line_match2(k,3)+size(img1,2); y2 = line_match2(k,4);
    plot([x1,x2], [y1,y2], 'LineWidth', 2, 'Color', [1, 0, 0]);
end
%% get more point matches based line matching
% if size(lpts1,2)>0
%     [matchtmppts_1, matchtmppts_2] = multiSample_APAP(lpts1, lpts2, img1, img2);
%     matchpts_1 = [matches_1, matchtmppts_1];
%     matchpts_2 = [matches_2, matchtmppts_2];
% else
%     matchpts_1 = [matches_1];
%     matchpts_2 = [matches_2];
% end
% deind1 = find(matchpts_1(1,:)>size(img1,2) | matchpts_1(2,:)>size(img1,1) | matchpts_1(1,:)<0 | matchpts_1(2,:)<0);
% matchpts_1(:,deind1) = [];
% matchpts_2(:,deind1) = [];
% deind2 = find(matchpts_2(1,:)>size(img1,2) | matchpts_2(2,:)>size(img1,1) | matchpts_2(1,:)<0 | matchpts_2(2,:)<0);
% matchpts_1(:,deind2) = [];
% matchpts_2(:,deind2) = [];

matchpts_1 = matches_1;
matchpts_2 = matches_2;

%% warping and blending
fprintf('> warping and blending...');tic;

% evaluate the homography based on the point-feature
if isempty(line_match1)
    pts_line_H = calcHomo(matchpts_1, matchpts_2);
% evaluate the homography based on the dual-feature
else
    [h, ~, T1, T2] = calcHomoPointLine( matchpts_1, matchpts_2, line_match1, line_match2 );
    pts_line_H = T2\(h*T1);
end
fprintf('done (%fs)\n',toc);

%% generating mesh grid (C1*C2) to optimize warped control vertices and rotation angle theta
[X, Y] = meshgrid(linspace(1,size(img1,2),C2+1), linspace(1,size(img1,1),C1+1)); % mesh grid index
% Mesh (cells) vertices' coordinates.
Mv = [X(:), Y(:)];
init_H = pts_line_H;
init_H = init_H./(init_H(end));
theta = atan2(-init_H(6), -init_H(3));  % evaluate the rotation (r.t. original coordinate system)

%% use rotation angle theta to calculate normal vector of warped v-line normal_vec
fprintf('> Generating u-v sample points and u-v term to optimize V*...');tic;
[lines_vs, lines_us, lines_ue] = generateUV( img1, img2, init_H, theta, C1 ,C2 ); % rotated vertical and horizontal lines
nor_vec_v = [init_H(2)*init_H(6)-init_H(5)*init_H(3), init_H(4)*init_H(3)-init_H(1)*init_H(6)]; % the normal vector of v-lines after transformation 
nor_vec_v = nor_vec_v./norm(nor_vec_v);  % normalization
sparse_v  = energyLineV( img1, C1, C2, lines_vs, nor_vec_v ); % energy of preserving v-lines
[sparse_us, sparse_ue] = energyLineU( img1, C1, C2, lines_us, lines_ue, init_H );  % energy of preserving u-slope and u-equidistant
fprintf('done (%fs)\n',toc);

%% calculate the new alignment energy term with scale operator
% 计算点对齐项和线对齐项
fprintf('> Generating scale-alignment term ||AV*-b||^2 to optimize V*...');tic;
[ sparse_point_align, psMatch ] = energyPointAlign( img1, C1, C2, matchpts_1, matchpts_2 );  % 点对齐项，根据特征点匹配
[ sparse_line_align, cMatch ] = energyLineAlign( img1, C1, C2, line_match1, line_match2 );  % 线对齐项，根据线特征匹配
fprintf('done (%fs)\n',toc);

%% calculate line-preserving term with line segments
% 单独重新提取左图的线段，并聚合生成长线段，划分局部线段和全局线段，计算局部线保持项和全局线保持项
fprintf('> Detect line segments and merge global lines ...');tic;
[sa_locallines, sl_locallines, sa_globallines, sl_globallines] = alllinesDetect( path1, img1, C1, C2 );  % detect longlines using hough line detection and merge long lines.
fprintf('done (%fs)\n', toc);

% % 不使用线保持项
% sa_locallines = [0,0,0,0,0,0,0];
% sl_locallines = 0;
% sa_globallines = zeros(1,57);
% sl_globallines = 0;

% sa:sample, sl:slope
fprintf('> Calculate line-preserving term to optimize V*...');tic;
[sparse_locallines, sparse_globallines] = energyLineSegments(img1, sa_locallines, sl_locallines, sa_globallines, sl_globallines, init_H, C1, C2);
fprintf('done (%fs)\n', toc);

%% construct matrix A,b to minimize ||Ax-b||^2  (A'*Ax=A'*b)
% E_warp = E_align + E_usample + E_vsample + E_line, 
zero_len = size(sparse_us,1)+size(sparse_v,1)+size(sparse_ue,1)+size(sparse_locallines,1)+size(sparse_globallines,1);   

Mv=Mv';
init_V = Mv(:);

para_g = parameters.grid_size;
para_p = parameters.point_align;
para_l = parameters.line_align;
para_ps = parameters.perspective;
para_pj = parameters.projective;
para_sl = parameters.locallines;
para_ll = parameters.globallines;
Matrix_A = [sqrt(para_p).*sparse_point_align; sqrt(para_l).*sparse_line_align; sqrt(para_ps).*sparse_us; sqrt(para_ps).*sparse_v;...
            sqrt(para_pj).*sparse_ue;  sqrt(para_sl).*sparse_locallines; sqrt(para_ll).*sparse_globallines]; 
m_x = [psMatch; sqrt(para_l).*cMatch; zeros(zero_len,1)]; 

%% use iterative methods in sparse linear system to solve the energy minimization
fprintf('> Use LSQR method to calcuate optimized V_star...');tic;
[V_star, flag, ~, iter] = lsqr(Matrix_A, m_x, 1e-8 , 5000, [], [], init_V);
fprintf('done (%fs)\n',toc);
optimized_V = vec2mat(V_star,2); clear V_star; % show the warped control vertices in axis(N*2 x,y)

% figure;
% hold on;
% for i=1:size(optimized_V,1)
%     plot(optimized_V(i,1),optimized_V(i,2),'ro','LineWidth',2);
% end

%% calculate the warp image using warp function (homography)
fprintf('> Mesh deformation using bilinear interpolation...');tic;
wX = reshape(optimized_V(:,1), C1+1, C2+1);
wY = reshape(optimized_V(:,2), C1+1, C2+1);
warped_img1 = meshmap_warp2homo(img1, X, Y, wX, wY); % image for RGB % 左图经过网格变形后的图像
fprintf('done (%fs)\n',toc);
% figure;
% imshow(warped_img1)

%% seek the optimal seam by graph-cut to blend the warped images, show the final result
% 用图割法寻找最佳缝合线来融合变形后的图像，展示最终结果
% Canvas size. % 画布尺寸，即最终输出图像尺寸
off = ceil([ 1 - min([1 optimized_V(:,1)']) + 1 ; 1 - min([1 optimized_V(:,2)']) + 1 ]);
cw = max([ceil(optimized_V(:,1))', size(img2,2)])+off(1)-1; % 画布宽度
ch = max([ceil(optimized_V(:,2))', size(img2,1)])+off(2)-1; % 画布高度

%% calculate the point-rmse and line-rmse for quantitative evaluation
% 计算点的RMSE和线的RMSE用于定量评价
% calcualte the rmse for points
rmse = RMSE(img1, C1, C2, matches_1, matches_2, wX, wY, off);
% calcualte the rmse for lines
[rmse_dis, rmse_err, rmse_cross] = RMSEline(img1, C1, C2, line_match1, line_match2, wX, wY, off);

%% draw the mesh of the target image
meshdraw(warped_img1, wX, wY, off, C1, C2); % 在变形后的左图上面画上变形后的网格

%% the warped image and the reference image
img1Homo = zeros(ch,cw,3); img2Homo = img1Homo; % 初始化两张和最终图像尺寸一致的全黑图像
% img1Homo：变形后的左图在最终拼接图像上的位置
% img2Homo：右图在最终拼接图像上的位置
img1Homo(floor(min(optimized_V(:,2)))+off(2)-1:floor(min(optimized_V(:,2)))+off(2)-2+size(warped_img1,1),...
    floor(min(optimized_V(:,1)))+off(1)-1:floor(min(optimized_V(:,1)))+off(1)-2+size(warped_img1,2), :) = warped_img1; 
img2Homo(off(2):(off(2)+size(img2,1)-1),off(1):(off(1)+size(img2,2)-1),:) = img2;

%% salient object detection
% 显著目标检测
% 输出图像为输入图像的每个像素属于前景的概率，值在0-1之间
pmap1 = mbs_saliency(warped_img1);
pmap2 = mbs_saliency(img2);

% 将显著目标检测结果放到最终拼接图像上的对应位置
homo_pmap1 = zeros(ch,cw);  homo_pmap2 = homo_pmap1;
homo_pmap1(floor(min(optimized_V(:,2)))+off(2)-1:floor(min(optimized_V(:,2)))+off(2)-2+size(pmap1,1),...
    floor(min(optimized_V(:,1)))+off(1)-1:floor(min(optimized_V(:,1)))+off(1)-2+size(pmap1,2), :) = pmap1; 
homo_pmap2(off(2):(off(2)+size(pmap2,1)-1),off(1):(off(1)+size(pmap2,2)-1),:) = pmap2;

%% graph-cut for image blending
% 用图割法计算缝合线
fprintf('> Seam cutting...');tic;
imgout = blendTexture(img1Homo, homo_pmap1, img2Homo, homo_pmap2);
% figure, imshow(imgout);
fprintf('done (%fs)\n',toc);

%% output
% 不转换黑白底色，直接保存结果
pngout = sprintf('%d-%d-%d-%d-%d-%d-%d-%s.jpg', para_g, para_p, para_l, para_ps, para_pj, para_sl, para_ll, datestr(now, 30)); 
imwrite(imgout, [outpath,pngout]);  % 不转换黑白底色
fprintf('RMSE: %f\n', rmse);
fprintf('RMSE error: %f\n', rmse_err);
fprintf('RMSE distance: %f\n', rmse_dis);
fprintf('RMSE crossproduct: %f\n', rmse_cross);
%% linear blending
off = ceil([ 1 - min([1 optimized_V(:,1)']) + 1 ; 1 - min([1 optimized_V(:,2)']) + 1 ]);
cw = max([ceil(optimized_V(:,1))', size(img2,2)])+off(1)-1;
ch = max([ceil(optimized_V(:,2))', size(img2,1)])+off(2)-1;

img1Homo = zeros(ch,cw,3); img2Homo = zeros(ch,cw,3);
img1Homo(floor(min(optimized_V(:,2)))+off(2)-1:floor(min(optimized_V(:,2)))+off(2)-2+size(warped_img1,1),...
    floor(min(optimized_V(:,1)))+off(1)-1:floor(min(optimized_V(:,1)))+off(1)-2+size(warped_img1,2), :) = warped_img1; 
img2Homo(off(2):(off(2)+size(img2,1)-1),off(1):(off(1)+size(img2,2)-1),:) = img2;

linear_out = imageBlending(img1Homo, img2Homo, 'linear');     
pngout = sprintf('linear-blending.png'); 
imwrite(linear_out, [outpath, pngout]);
    
fprintf('done (%fs)\n',toc);

