function [predict_label,score_s,model] = Bio_MixCorDeepzFuzzy_ESN(Train_Matrix_dis,Train_y,Test_vs_Train_Matrix_dis,options)

%
%
% options.lambda Regularization coefficient
% options.k: number of fuzzy rules

% options.gamma: width of kernel
%
%
seed = 12345678;
rand('seed', seed);
y_pred_test=[];
model = [];

V = options.L;

[n_tr,d] = size(Train_Matrix_dis);


%if-parts
ss=sum(Train_Matrix_dis);
[B I] = sort(ss,'descend');
Support_Bio_indexs = I(1:options.k);
Support_Bio_Train_s = Train_Matrix_dis(:,Support_Bio_indexs);
Support_Bio_Test_s = Test_vs_Train_Matrix_dis(:,Support_Bio_indexs);

g = options.gamma;

[G_train] = calc_mu_squ(Support_Bio_Train_s,g);
[G_test] = calc_mu_squ(Support_Bio_Test_s,g);






nInternalUnits = options.nInternalUnits;
nOutputUnits = size(Train_y,2);
[n_samples,nInputUnits] = size(G_train);
%construct networks

for v=1:V
	if v==1
	l_1 = nInputUnits;
	else
	l_1 = nInternalUnits(v-1);
	end
	model{v}.B_in = 2.0 * rand(nInternalUnits(v), l_1)'- 1.0;
	connectivity = min([10/nInternalUnits(v) 1]);
	model{v}.internalWeights_UnitSR = generate_internal_weights(nInternalUnits(v), ...
                                                connectivity);
																				
												
	model{v}.B_res = model{v}.internalWeights_UnitSR' * options.rho;
end

%Processing training data
All_S=[];
for v=1:V
	if v==1
	im_vectors = G_train;All_S=[All_S,G_train];
	else
	im_vectors = model{v-1}.S;
	end
	S = im_vectors*model{v}.B_in;
	S = tanh(S);
	temp_x = im_vectors*model{v}.B_in  + S*model{v}.B_res;
	new_S = tanh(temp_x);
	clear S;clear temp_x;

	model{v}.S=[new_S];
	All_S=[new_S,All_S];

end
	
	Alpha=rand(size(All_S,2),nOutputUnits);
	%model.nOutputUnits = nOutputUnits;
	
	I_lambda_res = eye(size(All_S,2));
	
	%model_Z = (options.lambda*I_lambda_res + All_S'*All_S)\(All_S'*Train_y);
%training
loss_list=[];
Sigma_list = rand(options.M,1)*1;
e_list=[];
c_err_list=[];
sigma_err_list=[];
h_err_list=[];
mu_new=[];sig_news=[];
for i=1:options.iterMax
	xii = All_S*Alpha - Train_y;
	e_list=[e_list,xii];
	[mu_new,sig_news] = update_sig(All_S,Train_y,Alpha,options);
	Sigma_list = sig_news;
	S_M = computing_SM(mu_new,Sigma_list,options);
	k_sigma_vector = computing_k_sigma(All_S,Train_y,Alpha,mu_new,Sigma_list);

	h = sum(k_sigma_vector,1)/n_tr;

	nu_v_l = (S_M + 0.05*eye(options.M))\h';
	loss_v = computing_loss(All_S,Train_y,Alpha,mu_new,Sigma_list,nu_v_l);loss_list=[loss_list,loss_v];
	
	theta_v = computing_theta(All_S,Train_y,Alpha,mu_new,Sigma_list,nu_v_l);
	theta_v2 = computing_theta2(All_S,Train_y,Alpha,mu_new,Sigma_list,nu_v_l);
	theta_v = 1./(theta_v+0.001);
	THETA = diag(theta_v);
	Alpha = (All_S'*THETA*All_S + 2*n_tr*options.lambda*I_lambda_res)\(All_S'*THETA*Train_y-All_S'*theta_v2);
	c_err_list=[c_err_list,mu_new];
	sigma_err_list=[sigma_err_list,Sigma_list];
	h_err_list=[h_err_list,h'];

end

model{V+1}.Z=Alpha;
	
model{V+1}.c_err_list = c_err_list;
model{V+1}.sigma_err_list = sigma_err_list;
model{V+1}.h_err_list = h_err_list;
model{V+1}.loss_list = loss_list;
model{V+1}.e_list = e_list;

All_S_test=[];
%predicting
for v=1:V
	if v==1
	im_vectors = G_test;All_S_test=[All_S_test,G_test];
	else
	im_vectors = model{v-1}.S;
	end
	
	S=im_vectors*model{v}.B_in;
	S = tanh(S);
	temp_x = im_vectors*model{v}.B_in+S*model{v}.B_res;
	
	new_S = tanh(temp_x);
	model{v}.S=[new_S];
	All_S_test=[new_S,All_S_test];
	%new_S = [tanh(temp_x)];

	clear temp_x; clear S;
end



	score_s = All_S_test*Alpha;
	predict_label = ones(size(All_S_test,1),1);
	tdst = options.tt;
	predict_label(find(score_s<tdst))=-1;
	
%predict_label = sign(score_s);

end


function [c_e, sig_news] = update_sig(X,Y,A,options)
	xi_1 = X*A - Y;
	[n_examples, d] = size(xi_1);
	
	
	[v,U,~] = fcm(xi_1,options.M,[2,NaN,1.0e-6,0]);

	for i=1:options.M
		v1 = repmat(v(i,:),n_examples,1);
		u = U(i,:);
		uu = repmat(u',1,d);
		b(i,:) = sum((xi_1-v1).^2.*uu,1)./sum(uu)./1;
	end
	sig_news = sqrt(b);
	c_e = v;
end


function Sigam_M = computing_SM(mu_list,Sigma_list,options)
	S_M = zeros(options.M,options.M);

for i=1:options.M
	for j=1:options.M
		temp_z = (mu_list(i)-mu_list(j))^2;
		temp_z = 0.5*temp_z/(Sigma_list(i)^2+Sigma_list(j)^2);
		temp_z = exp(-1*temp_z);
		S_M(i,j) = temp_z/(sqrt(Sigma_list(i)^2+Sigma_list(j)^2)); 
	end
end

	Sigam_M = S_M/(sqrt(2*3.14159));

end


function loss_v = computing_loss(X,Y,A,mu_l,sigma_l,nu_l)

k_sigma_vector = computing_k_sigma(X,Y,A,mu_l,sigma_l);
	temp_matrix = zeros(size(k_sigma_vector));
	
	for i=1:length(sigma_l)
		tt = nu_l(i)/(sigma_l(i)^2);
		temp_matrix(:,i) = tt.*k_sigma_vector(:,i);
	
	end
	loss_v=sum(sum(temp_matrix));
end



function theta_v = computing_theta(X,Y,A,mu_l,sigma_l,nu_l)
	
	k_sigma_vector = computing_k_sigma(X,Y,A,mu_l,sigma_l);
	temp_matrix = zeros(size(k_sigma_vector));
	
	for i=1:length(sigma_l)
		tt = nu_l(i)/(sigma_l(i)^2);
		temp_matrix(:,i) = tt.*k_sigma_vector(:,i);
	
	end
	theta_v=sum(temp_matrix,2);
	
end


function theta_v2 = computing_theta2(X,Y,A,mu_l,sigma_l,nu_l)
	
	k_sigma_vector = computing_k_sigma(X,Y,A,mu_l,sigma_l);
	temp_matrix = zeros(size(k_sigma_vector));
	
	for i=1:length(sigma_l)
		tt = (nu_l(i)*mu_l(i))/(sigma_l(i)^2);
		temp_matrix(:,i) = tt.*k_sigma_vector(:,i);
	
	end
	theta_v2=sum(temp_matrix,2);
	
end


function k_sigma_vector = computing_k_sigma(X,Y,A,mu_l,sigma_l)
	
	k_sigma_vector = zeros(size(X,1),length(sigma_l));
	xi_1 = X*A - Y;
	for i = 1:length(sigma_l)
	k_sigma_vector(:,i) = Guss_func(xi_1,mu_l(i),sigma_l(i));
	
	end
end


function Guss_v = Guss_func(x_temp,mu,sig)

	temp_1 = (x_temp-mu).^2;
	temp_1 = temp_1./(2*sig*sig);
	Guss_v = exp(-1*temp_1)/(sqrt(2*3.14159));
end