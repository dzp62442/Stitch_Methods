% Algorithm parameters

box1 = [1:size(im1,2)        1:size(im1,2)                    ones(1,size(im1,1))  size(im1,2)*ones(1,size(im1,1)) ;
        ones(1,size(im1,2))  size(im1,1)*ones(1,size(im1,2))  1:size(im1,1)        1:size(im1,1) ;
        ones(1,size(im1,2))  ones(1,size(im1,2))              ones(1,size(im1,1))  ones(1,size(im1,1)) ] ;
box2 = [1:size(im2,2)        1:size(im2,2)                    ones(1,size(im2,1))  size(im2,2)*ones(1,size(im2,1)) ;
        ones(1,size(im2,2))  size(im2,1)*ones(1,size(im2,2))  1:size(im2,1)        1:size(im2,1) ;
        ones(1,size(im2,2))  ones(1,size(im2,2))              ones(1,size(im2,1))  ones(1,size(im2,1)) ] ;

fe = max(M1(1,1), M2(1,1));
[box1_(1,:), box1_(2,:)] =  trans_fisheye2equi(box1(1,:), box1(2,:), eye(3), M1, D1, fe);
[box2_(1,:), box2_(2,:)] =  trans_fisheye2equi(box2(1,:), box2(2,:), inv(R+t*Nd_glb'), M2, D2, fe);

u0 = min([box1_(1,:), box2_(1,:)]);
u1 = max([box1_(1,:), box2_(1,:)]);
ur = u0:u1;
v0 = min([box1_(2,:), box2_(2,:)]);
v1 = max([box1_(2,:), box2_(2,:)]);
vr = v0:v1;
mosaicw = size(ur,2);
mosaich = size(vr,2);
offsetu = 1 - u0;
offsetv = 1 - v0;

margin = 0.1 * imsize2(2); % additional margin of the overlapping area
sub_u0 = max(min(box1_(1,:)), min(box2_(1,:)))-margin;
sub_u1 = min(max(box1_(1,:)), max(box2_(1,:)))+margin;
sub_v0 = max(min(box1_(2,:)), min(box2_(2,:)))-margin;
sub_v1 = min(max(box1_(2,:)), max(box2_(2,:)))+margin;
if margin > 0
    sub_u0 = max(u0,sub_u0);
    sub_u1 = min(u1,sub_u1);
    sub_v0 = max(v0,sub_v0);
    sub_v1 = min(v1,sub_v1);
end

[up,vp] = meshgrid(ur,vr) ;

[u, v] = trans_equi2fisheye(up, vp, eye(3), M1, D1, fe);

% triangular facet transformation
% merge the coincided points
ok_nd1 = false(size(X1_ori,2),1);
[~, idx1] = unique(round(X1_ori'), 'rows', 'stable');
ok_nd1(idx1) = true;
ok_nd2 = false(size(X2_ori,2),1);
[~, idx2] = unique(round(X2_ori'), 'rows', 'stable');
ok_nd2(idx2) = true;
ok_nd = ok_nd1 & ok_nd2;
X1_nd = X1_ori(:,ok_nd);
X2_nd = X2_ori(:,ok_nd);
n = sum(ok_nd);

nsq = ceil(sqrt(n));
deltau = (sub_u1-sub_u0)/nsq;
deltav = (sub_v1-sub_v0)/nsq;
u_add_lcl = [sub_u0:deltau:sub_u1-deltau, sub_u1*ones(1,nsq),          sub_u1:-deltau:sub_u0+deltau, sub_u0*ones(1,nsq)];
v_add_lcl = [sub_v0*ones(1,nsq),          sub_v0:deltav:sub_v1-deltav, sub_v1*ones(1,nsq),           sub_v1:-deltav:sub_v0+deltav];
u_add_glb = [u0, u1, u0, u1];
v_add_glb = [v0, v0, v1, v1];

[u_pano, v_pano] = trans_fisheye2equi(X1_nd(1,:), X1_nd(2,:), eye(3), M1, D1, fe);
[x1_add_lcl, y1_add_lcl] = trans_equi2fisheye(u_add_lcl, v_add_lcl, eye(3), M1, D1, fe);
X1_add_lcl = [x1_add_lcl; y1_add_lcl; ones(1,4*nsq)];
X2_add_lcl = zeros(size(X1_add_lcl));
Y1_nd = trans_fisheye2cam_X(X1_nd, M1, D1);
Y2_nd = trans_fisheye2cam_X(X2_nd, M2, D2);
for klcl = 1:size(X1_add_lcl,2)
    dist2 = (u_pano-u_add_lcl(klcl)).^2 + (v_pano-v_add_lcl(klcl)).^2;
    weights = exp(-dist2/(200*200));
    Nd_lcl = ndp_ransac(Y1_nd, Y2_nd, weights, R, t, 0, 0);
    
    [x2_add_lcl, y2_add_lcl] = trans_equi2fisheye(u_add_lcl(klcl), v_add_lcl(klcl), R+t*Nd_lcl', M2, D2, fe);
    X2_add_lcl(:,klcl) = [x2_add_lcl; y2_add_lcl; 1];
end
[x1_add_glb, y1_add_glb] = trans_equi2fisheye(u_add_glb, v_add_glb, eye(3), M1, D1, fe);
X1_add_glb = [x1_add_glb; y1_add_glb; ones(1,4)];
[x2_add_glb, y2_add_glb] = trans_equi2fisheye(u_add_glb, v_add_glb, R+t*Nd_glb', M1, D1, fe);
X2_add_glb = [x2_add_glb; y2_add_glb; ones(1,4)];

X1_nd = [X1_nd, X1_add_lcl, X1_add_glb];
X2_nd = [X2_nd, X2_add_lcl, X2_add_glb];

[xcam, ycam, zcam] = trans_fisheye2cam(X1_nd(1,:), X1_nd(2,:), M1, D1);
x_pano = xcam;
y_pano = ycam;
z_pano = zcam;
x1 = xcam ./ zcam;
y1 = ycam ./ zcam;
[xcam, ycam, zcam] = trans_fisheye2cam(X2_nd(1,:), X2_nd(2,:), M2, D2);
x2 = xcam ./ zcam;
y2 = ycam ./ zcam;

% triangulate
u_pano = [u_pano, u_add_lcl, u_add_glb];
v_pano = [v_pano, v_add_lcl, v_add_glb];
% constrained_edge = [n+1 n+2; n+2 n+4; n+4 n+3; n+3, n+1];
% tri = delaunayTriangulation([x_pano',y_pano'], constrained_edge);
% tri = delaunay([u_pano',v_pano']);
tri = convhulln([x_pano',y_pano', z_pano']);
tri = tri(tri(:,1) <= n+size(X1_add_lcl,2) | tri(:,2) <= n+size(X1_add_lcl,2) | tri(:,3) <= n+size(X1_add_lcl,2), :);
ntri = size(tri,1);

Nd_all = zeros(3, ntri);
u_ = zeros(mosaich, mosaicw);
v_ = zeros(mosaich, mosaicw);

for ktri = 1:ntri
    % original coordinates
    x1_tri = x1(tri(ktri,:));
    y1_tri = y1(tri(ktri,:));
    x2_tri = x2(tri(ktri,:));
    y2_tri = y2(tri(ktri,:));
    Y1_tri = [x1_tri; y1_tri; ones(1,3)];
    Y2_tri = [x2_tri; y2_tri; ones(1,3)];
    
    Nd_tri = ndp_ransac(Y1_tri, Y2_tri, ones(3,1), R, t, 0, 0);
    Nd_all(:,ktri) = Nd_tri;
    
     % coordinates in the panoramic image
    u_pano_tri = u_pano(tri(ktri,:));
    v_pano_tri = v_pano(tri(ktri,:));
    
    %set bounding box of current triangle
    ub0_offset = min(u_pano_tri) + offsetu;
    ub1_offset = max(u_pano_tri) + offsetu;
    vb0_offset = min(v_pano_tri) + offsetv;
    vb1_offset = max(v_pano_tri) + offsetv;
    margin_tri_u = 0.3 * (ub1_offset - ub0_offset);
    margin_tri_v = 0.3 * (vb1_offset - vb0_offset);
    ub0_offset = max(1, ceil(ub0_offset - margin_tri_u));
    ub1_offset = min(mosaicw, floor(ub1_offset + margin_tri_u));
    vb0_offset = max(1, ceil(vb0_offset - margin_tri_v));
    vb1_offset = min(mosaich, floor(vb1_offset + margin_tri_v));
    ub0 = ub0_offset - offsetu;
    ub1 = ub1_offset - offsetu;
    vb0 = vb0_offset - offsetv;
    vb1 = vb1_offset - offsetv;
    ubr = ub0:ub1;
    vbr = vb0:vb1;
    [uub,vvb] = meshgrid(ubr,vbr);
    boxw = size(ubr,2);
    boxh = size(vbr,2);
    if boxw == 0 || boxh == 0
        continue;
    end
    
    % find inliers of the current triangle (on the unit sphere)
    % scalar
    xtri = x_pano(tri(ktri,:));
    ytri = y_pano(tri(ktri,:));
    ztri = z_pano(tri(ktri,:));
    xOA = xtri(1);
    yOA = ytri(1);
    zOA = ztri(1);
    xAB = xtri(2) - xtri(1);
    yAB = ytri(2) - ytri(1);
    zAB = ztri(2) - ztri(1);
    xAC = xtri(3) - xtri(1);
    yAC = ytri(3) - ytri(1);
    zAC = ztri(3) - ztri(1);
    % array
    [xcam, ycam, zcam] = trans_equi2cam(uub, vvb, fe);
    detOAABAC = xAB*yAC*zcam - xAB*ycam*zAC - xAC*yAB*zcam + xAC*ycam*zAB + xcam*yAB*zAC - xcam*yAC*zAB;
    tOP = (yAB*zAC*xOA - yAC*zAB*xOA + xAC*zAB*yOA - xAB*zAC*yOA + xAB*yAC*zOA - xAC*yAB*zOA) * ones(boxh, boxw) ./ detOAABAC;
    uAB = -(yAC*zcam*xOA - ycam*zAC*xOA + xcam*zAC*yOA - xAC*zcam*yOA + xAC*ycam*zOA - xcam*yAC*zOA) ./ detOAABAC;
    vAC = -(ycam*zAB*xOA - yAB*zcam*xOA + xAB*zcam*yOA - xcam*zAB*yOA + xcam*yAB*zOA - xAB*ycam*zOA) ./ detOAABAC;
    
    trimask = (tOP >=0 & tOP <=1 & uAB >= 0 & vAC >= 0 & uAB + vAC <= 1);
    
    [ubox_, vbox_] = trans_equi2fisheye(uub, vvb, R+t*Nd_tri', M2, D2, fe);
    
    ubox_(~trimask) = 0;
    vbox_(~trimask) = 0;
    zeromask = (u_(vb0_offset:vb1_offset,ub0_offset:ub1_offset) == 0);
    u_(vb0_offset:vb1_offset,ub0_offset:ub1_offset) = u_(vb0_offset:vb1_offset,ub0_offset:ub1_offset) + zeromask .* ubox_;
    v_(vb0_offset:vb1_offset,ub0_offset:ub1_offset) = v_(vb0_offset:vb1_offset,ub0_offset:ub1_offset) + zeromask .* vbox_;
end

if exist('vl_imwbackward','file')
    im1_p = vl_imwbackward(im2double(im1), u, v) ;
else
    im1_p = zeros(mosaich,mosaicw,im_ch);
    for kc = 1:size(im1,3)
        im1_p(:,:,kc) = interp2(im2double(im1(:,:,kc)),u,v);
    end
end

if exist('vl_imwbackward','file')
    im2_p = vl_imwbackward(im2double(im2),u_,v_) ;
else
    im2_p = zeros(mosaich,mosaicw,im_ch);
    for kc = 1:size(im2,3)
        im2_p(:,:,kc) = interp2(im2double(im2(:,:,kc)),u_,v_);
    end
end

alpha1 = ones(size(im1_p,1),size(im1_p,2));
alpha1(isnan(im1_p(:,:,1)))=0;
alpha2 = ones(size(im2_p,1),size(im2_p,2));
alpha2(isnan(im2_p(:,:,1)))=0;

mask1 = ~isnan(im1_p);
mask2 = ~isnan(im2_p);
mass = mask1 + mask2 ;
im1_p(isnan(im1_p)) = 0 ;
im2_p(isnan(im2_p)) = 0 ;
mosaic = (im1_p + im2_p) ./ mass ;
% mosaic(mass==0) = 1;% white background

figure(6) ; clf ;
imshow(mosaic, 'border', 'tight') ; hold on;
plot(u_pano+offsetu, v_pano+offsetv, 'b.', 'MarkerSize', marker_size);
triplot(tri,u_pano+offsetu,v_pano+offsetv,'g');

mosaic(mass==0) = 1;% white background
figure(7) ; clf ;
imshow(mosaic, 'border', 'tight') ;


imwrite(mosaic, [exp_path 'mosaic_local.jpg']) ;