function [mu_m] = calc_mu_squ(M_score,b)




mu_m=[];

h_v = max(max(M_score));

ss = (M_score - h_v)/h_v;
ss = b*ss;

sim = exp(ss);



mu_m = sim;


end