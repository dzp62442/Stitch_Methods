subsetsize = 100;
subset = vl_colsubset(1:size(X1,2), subsetsize) ;
X1sub = X1(:,subset);
X2sub = X2(:,subset);

F = K2'\t_m*R/K1;
l = F'*X2sub;
l_ = F*X1sub;
% l = Fe'*X2sub(:,k);
% l_ = Fe*X1sub(:,k);

dx = 50.*abs(l(2,:))./sqrt(l(1,:).^2 + l(2,:).^2);
dx_ = 50.*abs(l_(2,:))./sqrt(l_(1,:).^2 + l_(2,:).^2);

figure(2); clf;
imshow([im1,im2], 'border', 'tight'); hold on;
plot(X1sub(1,:), X1sub(2,:), 'bo', 'MarkerSize', 10);
plot(X2sub(1,:)+imsize1(2), X2sub(2,:), 'bo', 'MarkerSize', 10);
x1 = X1sub(1,:) - dx;
x2 = X1sub(1,:) + dx;
y1 = (-l(3,:)-l(1,:).*x1)./l(2,:);
y2 = (-l(3,:)-l(1,:).*x2)./l(2,:);
line([x1; x2], [y1; y2], 'Color', 'g', 'LineWidth', 1.25);
x1_ = X2sub(1,:) - dx_;
x2_ = X2sub(1,:) + dx_;
y1_ = (-l_(3,:)-l_(1,:).*x1_)./l_(2,:);
y2_ = (-l_(3,:)-l_(1,:).*x2_)./l_(2,:);
line([x1_+imsize1(2); x2_+imsize1(2)], [y1_; y2_], 'Color', 'g', 'LineWidth', 1.25);