function [mu_m] = calc_mu(x,v,b)

% Calculate the X_g by x * fire-level

% x: the original data -- n_examples * n_features
% v: clustering centers of the fuzzy rule base -- k * n_features
% b: kernel width of the corresponding centers of the fuzzy rule base


n_examples = size(x,1);

[k,d] = size(v); % k: number of rules of TSK; d: number of dimensions
mu_m=[];
for i=1:k
	Y = v(i,:);
	r2 = repmat( sum(x.^2,2), 1, size(Y,1) ) ...
	+ repmat( sum(Y.^2,2), 1, size(x,1) )' ...
	- 2*x*Y' ;
	k = exp(-r2/b); 
	
   
    wt(:,i) = k;
end



mu_m = wt;


end