function [fimg1,fimg2] = MSE_EX(img,img2)
%HSI features
fimg1=EMAP(img,'', true, '', 'a', [100 200 500 1000],'s',[2.5 5 7.5 10]); 
fimg1=double(fimg1);
fimg1=Normalization(fimg1);
fimg1 = tsmooth(fimg1,0.002,4);
fimg1 =kpca(fimg1, 1000,round(size(fimg1,3).*0.8), 'Gaussian',20);%'Gaussian'
%LiDAR features
fimg2=EMAP(img2,'', true, '', 'a', [100 200 500 1000],'s',[2.5 5 7.5 10]); 
fimg2=double(fimg2);
fimg2=Normalization(fimg2);
fimg2 = tsmooth(fimg2,0.002,4);
fimg2 =kpca(fimg2, 1000,round(size(fimg2,3).*0.8), 'Gaussian',20);%'Gaussian'
end

