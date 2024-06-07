% v12 = fopen('temp50_D1_SQP.txtv12 = fopen('temp25_D1_IP_NOISY_SIGMA5_ACTUAL.txt','w');','w');
% v12 = fopen('May20_SQP_25AP_Sigma1_1_REalization.txt','w');
v12 = fopen('JUN6_SQPP_25AP_SigmaPOINT1_ACTUAL.txt','w');
% % % % % v12 = fopen('SQP_10_L2_MAY29_ACTUAL.txt','w');
% % % % % v12 = fopen('IP_10_L2_MAY29.txt','w');
%fopen('temp25_D1_SQP.txt','w'); v12 =
%fopen('2_6_IP_10APs_L2_ACTUAL.txt','w'); v12 =
%fopen('5_9_interiord_nosigma.txt','w');
for caus = 1:length(cdffff)-1
    
   fprintf(v12,'%0.8f, ', cdffff(caus));
    
end