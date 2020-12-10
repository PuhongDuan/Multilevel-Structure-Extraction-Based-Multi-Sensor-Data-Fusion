function w = subspace_classifier(x,y, lambda,MMiter)
%       
%
%  Subspace Multinomial Logistic Regression 
%  
%  see paper: 
%
%  Li, J.; Bioucas-Dias, J. M.; Plaza, A.; , "Spectral–Spatial Hyperspectral
%  Image Segmentation Using Subspace Multinomial Logistic Regression and 
%  Markov Random Fields," Geoscience and Remote Sensing, IEEE Transactions 
%  on , vol.PP, no.99, pp.1-15, 0  doi: 10.1109/TGRS.2011.2162649
%
% -- Input Parameters ----------------
%
% x ->      training set (each column represent a sample).
%           x can be samples or functions of samples, i.e., 
%           kernels, basis, etc.
% y ->      class (1,2,...,m)
% lambda -> sparsness parameter
% 
% MMiter -> Number of iterations (default 100)
%
% Author: Jun Li, Dec. 2011

if nargin == 3
    MMiter = 1000;
end
%[d - space dimension, n-number of samples]
[d,n] = size(x);
if (size(y,1) ~= 1) | (size(y,2) ~= n)
    error('Input vector y is not of size [1,%d]',n)
end

%[d - space dimension, n-number of samples]
[d,n] = size(x);
% number of classes 
m = max(y);

% auxiliar matrix to compute a bound for the logistic hessian
U = -1/2*(eye(m)-ones(m)/(m+1)); 
% convert y into binary information
Y = zeros(m,n);
for i=1:n
    Y(y(i),i) = 1;
end
% remove last line
% Y=Y(1:m-1,:);
% build Rx and B
nB = 2*m;
for i = 1:m
    x_k = [ones(1,n);x(i,:)];
    Rx(:,:,i) = x_k*x_k';
    B(:,:,i)=kron(U,Rx(:,:,i));
    dB(:,:,i) = B(:,:,i) - lambda*eye(nB);
end
w = eps+zeros(2,m);

%MMiter = 1000;
for i=1:MMiter

    % compute the  multinomial distributions (one per sample)
    p = [];
    L_k = [];
    for  i_k = 1:m
    aux_k = [];
    w_k = w(:,i_k);
    x_k = [ones(1,n);x(i_k,:)];
    aux_k = exp(w_k'*x_k);
    l_k = w_k'*x_k;
    p = [p;aux_k];
    L_k =[L_k;l_k];
    end
    
   
    p =  p./repmat(sum(p,1),m,1);
    
    for i_m = 1:m
        w_k = w(:,i_m);
        x_k = [ones(1,n);x(i_m,:)];
        Rx_k = Rx(:,:,i_m);
        dg = Rx_k*w_k*U(i_m,:)-x_k*(Y-p)';
        dg = dg(:);
%         w = w(:);    
        w_k = dB(:,:,i_m)\dg;
        w_k=reshape(w_k,2,m);
        w(:,i_m) = w_k(:,i_m);
    end
%     w=reshape(w,d,m);
    

end







