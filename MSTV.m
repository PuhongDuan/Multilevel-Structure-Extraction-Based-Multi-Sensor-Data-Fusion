function [ fimg ] = MSTV( img )
%% Spectral dimension Reduction
 img2=average_fusion(img,20);
 %%% size of image 
[no_lines, no_rows, no_bands] = size(img);

%% Normalization
no_bands=size(img2,3);
fimg=reshape(img2,[no_lines*no_rows no_bands]);
[fimg] = scale_new(fimg);
fimg=reshape(fimg,[no_lines no_rows no_bands]);
%% Structure extraction
 fimg1 = tsmooth(fimg,0.003,1);
 fimg2 = tsmooth(fimg,0.01,1);
 fimg3 = tsmooth(fimg,0.002,3);
 f_fimg=cat(3,fimg1,fimg2,fimg3);
%% Feature fusion with the KPCA
 fimg =kpca(f_fimg, 1000,30, 'Gaussian',20);%'Gaussian'


end

