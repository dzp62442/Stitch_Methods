function [Nd, ok, score ] = ndp_ransac(Y1, Y2, W, R, t, Nr, min_dis_y)
% ndp_ransac calculates the plane parameters based on the (weighted) point matches of two images.
% input:
%   Y1, Y2 are space coordinates on the unit projection plan of the two image, respectively.
%   W is the corresponding weights of the matched image pairs.
%   M1, M2 are instrinsic paremeter matrix
%   R, t are rotation matrix and translation vector from the first image to
%   the second one.
%   Nr is number of random trails to find the outliers.
%   min_dis_y: minimum ransac distance in the unit projection plan, <= 0 means no outliers would be
%   removed;
% output:
%   H: Homography
%   Nd = N/d, N is the estimated unit normal vector of the plane.
%             d is the distance between the plane and the optical center. (in the first camera coordinate system)
%   ok is the logical vector marking the inliers resulted from ransac.

N = size(Y1,2);
Y1(1,:) = Y1(1,:)./Y1(3,:);
Y1(2,:) = Y1(2,:)./Y1(3,:);
Y1(3,:) = ones(1,N);
Y2(1,:) = Y2(1,:)./Y2(3,:);
Y2(2,:) = Y2(2,:)./Y2(3,:);
Y2(3,:) = ones(1,N);

A = zeros(2*N, 3);
b = zeros(2*N, 1);

% t1=t(1); t2=t(2); t3=t(3);
% r1t=R(1,:); r2t=R(2,:); r3t=R(3,:);
% for i = 1:N
%     y1 = Y1(:,i);
%     y2 = Y2(:,i);
%     y1 = y1/y1(3);
%     y2 = y2/y2(3);
%     p2 = y2(1);
%     q2 = y2(2);
%     a1 = (p2*t3-t1) / (r3t*y1) * y1';
%     a2 = (q2*t3-t2) / (r3t*y1) * y1';
%     b1 = - p2 + (r1t*y1) / (r3t*y1);
%     b2 = - q2 + (r2t*y1) / (r3t*y1);
%     A(2*i-1:2*i,:) = W(i)*[a1;a2];
%     b(2*i-1:2*i) = W(i)*[b1;b2];
% end

A(1:2:2*N-1,1) = W(:) .* ((Y2(1,:)*t(3)-t(1)) ./ (R(3,:)*Y1) .* Y1(1,:))';
A(1:2:2*N-1,2) = W(:) .* ((Y2(1,:)*t(3)-t(1)) ./ (R(3,:)*Y1) .* Y1(2,:))';
A(1:2:2*N-1,3) = W(:) .* ((Y2(1,:)*t(3)-t(1)) ./ (R(3,:)*Y1) .* Y1(3,:))';
A(2:2:2*N,1) = W(:) .* ((Y2(2,:)*t(3)-t(2)) ./ (R(3,:)*Y1) .* Y1(1,:))';
A(2:2:2*N,2) = W(:) .* ((Y2(2,:)*t(3)-t(2)) ./ (R(3,:)*Y1) .* Y1(2,:))';
A(2:2:2*N,3) = W(:) .* ((Y2(2,:)*t(3)-t(2)) ./ (R(3,:)*Y1) .* Y1(3,:))';
b(1:2:2*N-1) = W(:) .* (- Y2(1,:) + (R(1,:)*Y1) ./ (R(3,:)*Y1))';
b(2:2:2*N) = W(:) .* (- Y2(2,:) + (R(2,:)*Y1) ./ (R(3,:)*Y1))';

if (Nr > 0 && min_dis_y > 0)
%     Nr = 100;
    Nd = zeros(3,Nr);
    ok = false(N,Nr);
    score = zeros(Nr,1);
    for ti = 1:Nr
        % estimate homograpyh
        subset = vl_colsubset(1:N, 3) ;
        subset_idx = [2*subset-1; 2*subset];
        subset_idx = subset_idx(:);
        
        if (rank(A(subset_idx, :)) < 3)
            ok(:,ti) = false(N,1);
            score(ti) = 0;
            continue;
        end
        Nd(:,ti) = A(subset_idx,:) \ b(subset_idx);
        
        % score homography
        duv = b-A*Nd(:,ti);
        du = duv(1:2:end);
        dv = duv(2:2:end);
        ok(:,ti) = (du.*du + dv.*dv) < min_dis_y*min_dis_y;
        score(ti) = sum(ok(:,ti).*W) ;
        
%         if (sum(ok(:,ti)) < 3)
%             ok(:,ti) = false(N,1);
%             score(ti) = 0;
%             continue;
%         end
    end
    [score, best] = max(score);
    ok = ok(:,best);
    
    idx = 1:N;
    idx = idx(ok);
    idx = [2*idx-1; 2*idx];
    idx = idx(:);
        
    Nd = A(idx,:) \ b(idx);
    
else
    Nd = A \ b;
    ok = true(N,1);
end

end
