function [ errH ] = residual_H_fisheye( X1, X2, K1, K2, R)

Y1 = trans_fisheye2cam_X(X1, K1, 0);
Y2 = trans_fisheye2cam_X(X2, K2, 0);

Y1_p2 = R * Y1;
Y1_p2(1,:) = Y1_p2(1,:) ./ Y1_p2(3,:) ;
Y1_p2(2,:) = Y1_p2(2,:) ./ Y1_p2(3,:) ;
errH1_2 = K2 * (Y2 - Y1_p2);
Y2_p1 = R' * Y2;
Y2_p1(1,:) = Y2_p1(1,:) ./ Y2_p1(3,:) ;
Y2_p1(2,:) = Y2_p1(2,:) ./ Y2_p1(3,:) ;
errH2_1 = K1 * (Y1 - Y2_p1);

errH = [errH1_2(1,:)', errH1_2(2,:)', errH2_1(1,:)', errH2_1(2,:)'];

end

