% RANSAC filtering with the global homography matrix
Nr_ndp = 1000; % Number of random trials for finding outliers
% min_dist_ndp = 50.0; % Distance threshold in pixels for finding outliers
Y1 = trans_fisheye2cam_X(X1, M1, D1);
Y2 = trans_fisheye2cam_X(X2, M2, D2);
% ransac_threshold = min_dist_ndp / max(M1(1,1), M2(1,1));

[Nd_glb, ok_ndp] = ndp_ransac(Y1, Y2, ones(size(Y1,2),1), R, t, Nr_ndp, ransac_threshold);

[E, ok_em] = EM_ransac_Y(Y1, Y2, 1000, 0.02);
ok_r = ok_ndp & ok_em;


figure(3); clf;
marker_size = 15;
imshow([im1,im2], 'border', 'tight'); hold on;
plot(X1(1,~ok_r), X1(2,~ok_r), 'r.', 'MarkerSize', marker_size);
plot(X1(1,ok_r), X1(2,ok_r), 'b.', 'MarkerSize', marker_size);

plot(X2(1,~ok_r)+imsize1(2), X2(2,~ok_r), 'r.', 'MarkerSize', marker_size);
plot(X2(1,ok_r)+imsize1(2), X2(2,ok_r), 'b.', 'MarkerSize', marker_size);

% h = line([X1(1,ok_ndp) ; X2(1,ok_ndp)+imsize1(2)], [X1(2,ok_ndp) ; X2(2,ok_ndp)], 'LineWidth', 2) ;

ok_count = sum(ok_r);
X1_ori = X1(:,ok_r);
X2_ori = X2(:,ok_r);