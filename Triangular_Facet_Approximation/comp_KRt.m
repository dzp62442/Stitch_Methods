% bundle adjustment on the normallized matching data
lambda = 0.01;
sigma_F = 1000;
sigma_H = 1000;
options = optimoptions('lsqnonlin', 'Algorithm','levenberg-marquardt', 'Display','final',...
    'MaxFunEvals',1e4, 'MaxIter',1e3, 'TolFun',1e-8, 'TolX',1e-8);
lb = [max(imsize1)/pi max(imsize2)/pi -pi -pi -pi -pi -pi];
ub = [Inf Inf pi pi pi pi pi];
[paras_init,resnorm_init,residual_init,exitflag_init,output_init] = lsqnonlin(...
    @(p)residual_KRt_robust(double(X1), double(X2), imsize1, imsize2, p, sigma_F, sigma_H, lambda), [1000 1000 0 0 0 0 0],...
    lb,ub,options);
r_norm = sqrt(residual_init(:,1).^2 + residual_init(:,2).^2 + residual_init(:,3).^2);
ok_lsq = (r_norm < 3 * mean(r_norm));
X1_ok = X1(:,ok_lsq);
X2_ok = X2(:,ok_lsq);

sigma_F = 1;
sigma_H = 1;
[paras,resnorm,residual,exitflag,output] = lsqnonlin(...
    @(p)residual_KRt_robust(double(X1_ok), double(X2_ok), imsize1, imsize2, p, sigma_F, sigma_H, lambda), paras_init,...
    lb,ub,options);
k1 = paras(1);
k2 = paras(2);
theta = paras(3:5);
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
t = [cos(phi_t)*cos(theta_t);cos(phi_t)*sin(theta_t);sin(phi_t)];

t_m = [0     -t(3) t(2)
       t(3)  0     -t(1)
       -t(2) t(1)  0];
M1 = K1; M2 = K2; D1 = 0; D2 = 0;

TestFM;