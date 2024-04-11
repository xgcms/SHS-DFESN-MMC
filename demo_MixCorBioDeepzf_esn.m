clear
seed = 12345678;
rand('seed', seed);
nfolds = 10; 


dataname = 'AAP.mat';options.lambda=0.0001;options.gamma=2^-4;options.k=130;options.nInternalUnits = [50,50,50];options.rho=0.1;options.L=3;options.iterMax = 2;options.M = 4; train_num = 214; options.tt = 0.0;
%dataname = 'ABP.mat';options.lambda=0.0001;options.gamma=2^-4;options.k=330;options.nInternalUnits = [100,100,100];options.rho=0.1;options.L=3;options.iterMax = 2;options.M = 3; train_num = 1600; options.tt = 0.0;
%dataname = 'ACP.mat';options.lambda=0.0001;options.gamma=2^-4;options.k=330;options.nInternalUnits = [100,50,50];options.rho=0.1;options.L=3;options.iterMax = 2;options.M = 2; train_num = 500;options.tt = 0.0;
%dataname = 'AIP.mat';options.lambda=0.0001;options.gamma=2^-4;options.k=780;options.nInternalUnits = [50,50,50];options.rho=0.1;options.L=3;options.iterMax = 3;options.M = 2; train_num = 3145; options.tt = -0.2;
%dataname = 'AVP.mat';options.lambda=0.0001;options.gamma=2^-2;options.k=330;options.nInternalUnits = [50,50,50];options.rho=0.1;options.L=3;options.iterMax = 4;options.M = 3; train_num = 951;options.tt = 0.0;

%dataname = 'CPP.mat';options.lambda=0.001;options.gamma=2^-5;options.k=220;options.nInternalUnits = [100,100,100];options.rho=0.1;options.L=3;options.iterMax = 2;options.M = 3; train_num = 740; options.tt = 0.0;
%dataname = 'QSP.mat';options.lambda=0.001;options.gamma=2^-3;options.k=210;options.nInternalUnits = [100,100,100];options.rho=0.1;options.L=3;options.iterMax = 2;options.M = 5; train_num = 400; options.tt = 0.0;
%dataname = 'SBP.mat';options.lambda=0.001;options.gamma=2^-3;options.k=110;options.nInternalUnits = [100,100,100];options.rho=0.1;options.L=3;options.iterMax = 2;options.M = 3; train_num = 160; options.tt = 0.2;

dataname
load(dataname);






labels(find(labels==0))=-1;
X = M_S(1:train_num,1:train_num);
y = labels(1:train_num);
ACC=[];SN=[];Spec=[];PE=[];NPV=[];F_score=[];MCC=[];AUC_list=[];
X = Knormalized(X);


score_s=[];
KP = 1:1:length(y);
crossval_idx = crossvalind('Kfold',KP,nfolds);

X_Y_test_label=[];
X_Y_dis=[];
for fold=1:nfolds
 
 train_idx = find(crossval_idx~=fold);
 test_idx  = find(crossval_idx==fold);
 
 
 K_Train = X(train_idx,train_idx);
 train_y = y(train_idx,1);
 
 K_Test = X(test_idx,test_idx);
 test_y = y(test_idx,1);
 
 K_test_Train = X(test_idx,train_idx);



	
	[predict_y,score_s,Alpha] = Bio_MixCorDeepzFuzzy_ESN(K_Train,train_y,K_test_Train,options);
	


		

 [ACC_i,SN_i,Spec_i,PE_i,NPV_i,F_score_i,MCC_i] = roc( predict_y,test_y );
 ACC=[ACC,ACC_i];SN=[SN,SN_i];Spec=[Spec,Spec_i];PE=[PE,PE_i];NPV=[NPV,NPV_i];F_score=[F_score,F_score_i];MCC=[MCC,MCC_i];
 
 X_Y_test_label=[X_Y_test_label;test_y];
X_Y_dis=[X_Y_dis;score_s(:,1)];
 [X_point,Y_point,THRE,AUC_i,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(test_y,score_s(:,1),1);
		fprintf('- FOLD %d - ACC: %f \n', fold, ACC_i)
		%break;
		AUC_list = [AUC_list,AUC_i];
 end
 
 [X_points,Y_points,THRE,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(X_Y_test_label,X_Y_dis,1);
 
mean_acc=mean(ACC)
mean_sn=mean(SN)
mean_sp=mean(Spec)
mean_mcc=mean(MCC)
 AUC
 mean_AUC = mean(AUC_list)