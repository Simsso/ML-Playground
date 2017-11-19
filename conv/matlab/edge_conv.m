I = imread('/Users/Denk/Pictures/social-m3-meta.jpg');
%I = rgb2gray(I);
I_r = I(:,2:end);
I_l = I(:,1:end-1);
I_t = I(1:end-1,:);
I_b = I(2:end,:);
I_conv_h = abs(I_l-I_r)+abs(I_r-I_l);
I_conv_v = abs(I_t-I_b)+abs(I_b-I_t);
I_out = max(I_conv_h(1:end-1,:),I_conv_v(:,1:end-1));
%I_out = imcomplement(I_out);
%I_conv = imbinarize(I_conv, .4);
imshow(I_out);
I_cropped = I(1:end-1,1:end-1);
imwrite([I_out], '/Users/Denk/Pictures/conv-out-tesla.jpg')