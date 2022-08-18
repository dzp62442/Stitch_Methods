function [ E, ok, score ] = EM_ransac_Y(Y1, Y2, Nr, min_dis)
N = size(Y1,2);

u = Y1(1,:)' ./ Y1(3,:)';
v = Y1(2,:)' ./ Y1(3,:)';
u_ = Y2(1,:)' ./ Y2(3,:)';
v_ = Y2(2,:)' ./ Y2(3,:)';

k = 1 ./ sqrt(u.*u+v.*v) + 1 ./ sqrt(u_.*u_+v_.*v_);

% coefficient matrix
A = [k.*u_.*u, k.*u_.*v, k.*u_, k.*v_.*u, k.*v_.*v, k.*v_, k.*u, k.*v, k];

if (min_dis > 0)
    E = cell(Nr);
    ok = cell(Nr);
    score = zeros(Nr,1);
    for t = 1:Nr
        % estimate foundamental matrix
        subset = vl_colsubset(1:N, 8) ;
        [U,S,V] = svd(A(subset,:));
        e = V(:,9);
        E_raw = reshape(e,3,3)';
        [U,S,V] = svd(E_raw);
%         disp([S(1,1),S(2,2),S(3,3)])

        S(1,1) = (S(1,1)+S(2,2))/2;
        S(2,2) = S(1,1);
        S(3,3) = 0;
        E{t} = U*S*V';

        % score foundamental matrix
        e = reshape(E{t}',9,1);
        dis = A * e;
        ok{t} = abs(dis) < min_dis;
        score(t) = sum(ok{t}) ;
        
    end
    [score, best] = max(score) ;
    ok = ok{best} ;
    [U,S,V] = svd(A(ok,:),'econ');
    e = V(:,9);
    E_raw = reshape(e,3,3)';
    [U,S,V] = svd(E_raw);
    S(1,1) = (S(1,1)+S(2,2))/2;
    S(2,2) = S(1,1);
    S(3,3) = 0;
    E = U*S*V';
else
    [U,S,V] = svd(A,'econ');
    e = V(:,9);
    E_raw = reshape(e,3,3)';
    [U,S,V] = svd(E_raw);
    S(3,3) = 0;
    E = U*S*V;
end

end