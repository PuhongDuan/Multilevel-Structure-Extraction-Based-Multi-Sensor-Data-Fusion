clc;
clear;
% close all;

addpath(genpath('AP'));
addpath(genpath('splitBregmanROF_mex'));
addpath(genpath('GraphCutMex'));

%% input your HS data
load Italy
img=double(HSI);
img2=LiDAR;
img=Normalization(img);
img2=Normalization(img2);
%% size of image 
[no_lines, no_rows, no_bands] = size(img);

tic;
%% Multilevel structure extraction
[fimg1,fimg2] = MSE_EX(img,img2);
%% Feature Fusion
 fea=cat(3,fimg1,fimg2);
 Feature=OTVCA_V3(fea,30);
%% classifier
[ class_HSI,p_HSI ] = MLR( Feature,Te,Tr );
mu=2;
no_classes=max(Te(:));
te=(log(p_HSI+eps))';
Dc = reshape(te,[no_lines, no_rows, 6]);%6 8 15 20
Sc = ones(no_classes) - eye(no_classes);
gch = GraphCut('open', -Dc, mu*Sc);
[gch MLRglmllmap] = GraphCut('expand',gch);
gch = GraphCut('close', gch);

%%
test_SL=matricetotwo(Te);
test_labels = test_SL(2,:)';
Result=MLRglmllmap(:)+1;
% Evaluation
GroudTest = double(test_labels(:,1));
ResultTest = Result(test_SL(1,:),:);
[SVM_OA,SVM_AA,SVM_Kappa,SVM_CA]=confusion(GroudTest,ResultTest);
testing_time=toc;
Result = reshape(Result,no_lines,no_rows);
VClassMap=label2colord(Result,'hu');
figure,imshow(VClassMap);
disp('%%%%%%%%%%%%%%%%%%% Classification Results of Fusion Method %%%%%%%%%%%%%%%%')
disp(['OA',' = ',num2str(SVM_OA),' ||  ','AA',' = ',num2str(SVM_AA),'  ||  ','Kappa',' = ',num2str(SVM_Kappa)])
