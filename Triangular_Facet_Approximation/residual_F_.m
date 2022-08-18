function [ errF ] = residual_F_( X1, X2, K1, K2, R, t )

t_m = [0     -t(3) t(2)
       t(3)  0     -t(1)
       -t(2) t(1)  0];

Y1 = trans_persp2cam_X(X1, K1, 0);
Y2 = trans_persp2cam_X(X2, K2, 0);
E = t_m*R;

l1_2 = E * Y1;
errF1_2 = Y2(1,:) .* l1_2(1,:) + Y2(2,:) .* l1_2(2,:) + Y2(3,:) .* l1_2(3,:);
s1_2 = 1 ./ sqrt((l1_2(1,:)/K2(1,1)).^2 + (l1_2(2,:)/K2(2,2)).^2);
errF1_2 = s1_2 .* errF1_2;
l2_1 = E' * Y2;
errF2_1 = Y1(1,:) .* l2_1(1,:) + Y1(2,:) .* l2_1(2,:) + Y1(3,:) .* l2_1(3,:);
s2_1 = 1 ./ sqrt((l2_1(1,:)/K1(1,1)).^2 + (l2_1(2,:)/K1(2,2)).^2);
errF2_1 = s2_1 .* errF2_1;

errF = [errF1_2', errF2_1'];

end

