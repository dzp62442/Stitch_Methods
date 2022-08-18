%% REW_palace
% %{
imfolder = 'images\REW_palace';
im_n = 24;
imfile = cell(im_n,1);
for ii = 1:im_n
    imfile{ii} = sprintf('%s\\%02d.jpg', imfolder, ii);
end

im = cell(im_n,1);
for ii = 1:im_n
    im{ii} = imread(imfile{ii});
end

edge_list = zeros(im_n-1,2);
for ei = 1:im_n-1
    edge_list(ei,:) = [ei,ei+1];
end
edge_list = cat(1, edge_list, [12,1; 24,13; 1,14; 2,15; 3,16; 4,17; 5,18; 6,19; 7,20; 8,21; 9,22; 10,23; 11,24]);

imsize = zeros(im_n,3);

for ii = 1:im_n
    imsize(ii,:) = size(im{ii});
    if imsize(ii,1) > 720
        scale = 720/size(im{ii}, 1);
        im{ii} = imresize(im{ii}, scale);

        imsize(ii,:) = size(im{ii});
    end
end

mosaic = REW_mosaic( im, edge_list, 0, 'equi', 0.02, imfolder );
%}
