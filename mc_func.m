function kv=mc_func(e,lammda_l,c_l,sigma_l)
	k_sigma_vector = zeros(length(e),length(sigma_l));
	for i = 1:length(sigma_l)
	k_sigma_vector(:,i) = Guss_func(e,c_l(i),sigma_l(i));
	
	end
	
	kv = k_sigma_vector*lammda_l;

end


function Guss_v = Guss_func(x_temp,mu,sig)

	temp_1 = (x_temp-mu).^2;
	temp_1 = temp_1./(2*sig*sig);
	Guss_v = exp(-1*temp_1)/(sqrt(2*3.14159));
end