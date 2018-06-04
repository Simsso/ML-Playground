I = imread('./tesla.jpg');
I = rgb2gray(I);
[height, width] = size(I)
xy_offset = 20
parts = 50
I_part = I(height/parts*xy_offset:height/parts*(xy_offset + 1),width/parts*xy_offset:width/parts*(xy_offset + 1));
C = xcorr2(I, I_part);
[C_height, C_width] = size(C)
C = (mat2gray(C) - ones(C_height, C_width)) * (-1);
imshow(C)
imwrite([C], './tesla-out.jpg');
