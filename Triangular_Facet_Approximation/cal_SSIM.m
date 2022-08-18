mask_overlap = mask1 & mask2;
bounds = bounding_box(mask_overlap);

im1_overlap = im1_p;
im1_overlap(~mask_overlap) = 0;
im1_overlap = im1_overlap(bounds(1):bounds(2),bounds(3):bounds(4),:);
im2_overlap = im2_p;
im2_overlap(~mask_overlap) = 0;
im2_overlap = im2_overlap(bounds(1):bounds(2),bounds(3):bounds(4),:);

SSIM = ssim(im2uint8(im1_overlap),im2uint8(im2_overlap));
disp(['SSIM = ',num2str(SSIM)]);