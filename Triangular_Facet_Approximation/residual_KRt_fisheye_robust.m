function [ err ] = residual_KRt_fisheye_robust( X1, X2, imsize1, imsize2, paras, sigma_F, sigma_H, lambda )

% parameretes
% sigma_F indicates the distance scope for inliers with fundamental matrix, in pixels
% sigma_H indicates the distance scope for inliers with homopraphy matrix, in pixels

% intrinsic parameters
k1 = paras(1);
k2 = paras(2);

% extrinsic parameters
theta = paras(3:5);
% yaw = paras(3);
% pitch = paras(4);
% roll = paras(5);
theta_t = paras(6);
phi_t = paras(7);

K1 = [k1, 0, imsize1(2)/2;
     0, k1, imsize1(1)/2;
     0,  0, 1];
K2 = [k2, 0, imsize2(2)/2;
      0, k2, imsize2(1)/2;
      0,  0, 1];
theta_m = [0         -theta(3) theta(2)
           theta(3)  0         -theta(1)
           -theta(2) theta(1)  0];
R = expm(theta_m);
% Ry = [cos(yaw),     0,              -sin(yaw);
%       0,            1,              0;
%       sin(yaw),     0,              cos(yaw)] ;
% Rp = [1,            0               0;
%       0,            cos(pitch),     sin(pitch);
%       0,            -sin(pitch),    cos(pitch)] ;
% Rr = [cos(roll),    sin(roll),      0;
%       -sin(roll),   cos(roll),      0;
%       0,            0,              1] ;
% R = Rr * Rp * Ry;
t = [cos(phi_t)*cos(theta_t);cos(phi_t)*sin(theta_t);sin(phi_t)];

errF = residual_F_fisheye( X1, X2, K1, K2, R, t );
% robust error function with outliers
outlier_F = (abs(errF) > sigma_F);
% errF(outlier_F) = sign(errF(outlier_F)) .* (sigma_F + sigma_F * log(abs(errF(outlier_F))/sigma_F));
errF(outlier_F) = sign(errF(outlier_F)) .* sqrt(2*sigma_F*abs(errF(outlier_F)) - sigma_F*sigma_F);

% homopraphy matrix

errH = residual_H_fisheye( X1, X2, K1, K2, R );
% robust error function with outliers
outlier_H = (abs(errH) > sigma_H);
% errH(outlier_H) = sign(errH(outlier_H)) .* (sigma_H + sigma_H * log(abs(errH(outlier_H))/sigma_H));
errH(outlier_H) = sign(errH(outlier_H)) .* sqrt(2*sigma_H*abs(errH(outlier_H)) - sigma_H*sigma_H);

err = [errF, sqrt(lambda) * errH];


end

