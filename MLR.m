function [ class,p ] = MLR( img,Te,Tr )
[row,col,no_band]=size(img);
im=reshape(img,[row*col no_band])';
% im = ToVector(img);
% im = im';
% spatial prior parameter
beta    = 2.0;
ksub =round( no_band.*0.8);
[l_all n_all]= size(im);
% 
% test_SL=matricetotwo(Te);
% test_samples = fimg2(:,test_SL(1,:))';
% test_labels = test_SL(2,:)';


 train1=matricetotwo(Tr);
%  train1 = trainall(:,indexes);%train1:2*1036
   y=train1(2,:);
    train = im(:,train1(1,:));%train:200*1036
 indexes=train1(1,:)';
  n_train = size(indexes,1);
[v,d]   = eig(train*train'/length(indexes));
 d = diag(d);
 %
 dtrue = d(ksub);
    
    % the remaining used for test
%  test1 = trainall;%test1:2*9330
%     test1(:,indexes) = [];  
test1=matricetotwo(Te);
 no_classes=max(Te(:));

%%  subspace classifier
 g = [];
  g_all= [];
  
  % number of classes
 
        for k_iter = 1:no_classes
            index_k = train1(2,:) == k_iter;
            train_k = train(:,index_k);     
            n_k   = size(train_k,2);
            [v,d]   = eig(train_k*train_k'/n_k);
            d = diag(d);
         
            sub_sp = d<dtrue;
            sub(k_iter) = sum(sub_sp);
            tau = sub(k_iter);
            
            
            P = v(:,1:tau)*v(:,1:tau)'*train;
            gall = zeros(1,n_train);
            
            for num_k = 1:n_train
                gall(num_k) = sqrt(P(:,num_k)'*P(:,num_k));
            end
            g = [g; gall];
            
            ggall=zeros(1,n_all);

            P_all = v(:,1:tau)*v(:,1:tau)'*im;
            
            for iter_all = 1:n_all
                ggall(iter_all) = sqrt(P_all(:,iter_all)'*P_all(:,iter_all));
            end
            g_all = [g_all;ggall];
        end
        
        
        % regularization parameter
        lambda = eps;
        w = subspace_classifier(-g,y,lambda);

        % compute the probablity
        p= subspace_mlogistic(w,-g_all);
         [maxp,class] = max(p);

end
