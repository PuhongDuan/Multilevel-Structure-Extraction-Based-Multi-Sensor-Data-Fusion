function p= subspace_mlogistic(w,x)

% compute the  multinomial distributions (one per sample)
m = size(w,2);
n = size(x,2);

p = [];
for  i = 1:m
    aux_k = [];
    w_k = w(:,i);
    x_k = [ones(1,n);x(i,:)];
    aux_k = exp(w_k'*x_k);
    p = [p;aux_k];
end
    p =  p./repmat(sum(p,1),m,1);

