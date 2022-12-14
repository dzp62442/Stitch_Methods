function S = ransac_global_similarity(data,data_orig,img1,img2)
thr_l = 0.1;
M = 500;

fprintf('  Showing results of RANSAC...');tic;
figure;
imshow([img1 img2]);
hold on;
for i=1:length(data_orig)
    plot(data_orig(1,i),data_orig(2,i),'go','LineWidth',2);
    plot(data_orig(4,i)+size(img1,2),data_orig(5,i),'go','LineWidth',2);
    plot([data_orig(1,i) data_orig(4,i)+size(img1,2)],[data_orig(2,i) data_orig(5,i)],'g-');
end
title('Ransac''s results');
fprintf('done (%fs)\n',toc);
pause(0.5)

for i = 1:20
    [ ~,res,~,~ ] = multigsSampling(100,data,M,10);
    con = sum(res<=thr_l);
    [ ~, maxinx ] = max(con);
    inliers = find(res(:,maxinx)<=thr_l);
    
    inliers = 1:1:length(data);  % 所有点均为内点

    if size(inliers) < 50
        break;
    end
    data_inliers = data(:,inliers);

    x  = data_inliers(1,:); 
    y  = data_inliers(2,:); 
    x_ = data_inliers(4,:);     
    y_ = data_inliers(5,:);
    
    A = [];
    b = [];
    
    for idx = 1:size(x,2)
        A = [A; x(idx) -y(idx) 1 0;
                y(idx)  x(idx) 0 1];

        b = [b;x_(idx);
               y_(idx)];
    end
    beta = A\b;
    
    S_segment{i} = [beta(1) -beta(2) beta(3);
                    beta(2)  beta(1) beta(4);
                         0        0       1];
    theta(i)     = atan(beta(2)/beta(1));
    
    clr = [rand(),0,rand()];
    plot(data_orig(1,inliers),data_orig(2,inliers),...
         'o','color',clr,'LineWidth',2);
    plot(data_orig(4,inliers)+size(img1,2),data_orig(5,inliers),...
         'o','color',clr,'LineWidth',2);
    hold on;
    pause(0.5);

    outliers = find(res(:,maxinx)>thr_l);
    data = data(:,outliers);
    data_orig = data_orig(:,outliers);
end

index = find(abs(theta) == min(abs(theta)));
S = S_segment{index};
end
