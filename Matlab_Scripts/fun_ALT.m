function [latency] = fun_ALT(alphasss,per_user_interference,all_states,L)
beta1 = 1; %keep it 10 for all channels
beta2 = 1;
S_u= 1*10^3;
B = 20*10^6; %bandwidth
f =  1*10^6;
C = 20;
p = 100;
ada_1 = 1;

% % if any((B*log2(1+((p*ada_1*alphasss.^2.*all_states.^2)./(1+per_user_interference))))==0)
% %     quit()
% % end
% % 
% % if any(isinf(max(beta2*((alphasss*S_u)./(B*log2(1+((p*ada_1*alphasss.^2.*all_states.^2)./(1+per_user_interference))))))))==1 ||...
% %     any(isnan(max(beta2*((alphasss*S_u)./(B*log2(1+((p*ada_1*alphasss.^2.*all_states.^2)./(1+per_user_interference))))))))==1    
% %    quit()
% % end
% % 

comp_latency = max((beta1*(sum(((alphasss(1:L,:).*S_u*C)/f),2))));
trans_latencies = [];

for row_idx = 1:6
    
    
    
    curr_alphasss = alphasss(1:L,row_idx);
    curr_etas = alphasss(L+1:2*L,row_idx);
    curr_all_states = all_states(:,row_idx);
    curr_per_user_interference = per_user_interference(:,row_idx);
    
    % Rate = (B*log2(1+((p*ada_1*curr_alphasss.^2.*curr_all_states.^2)./(1+curr_per_user_interference))));
    Rate = (B*log2(1+((p*curr_etas.*curr_all_states.^2)./(1+curr_per_user_interference))));
    curr_lat = max(beta2*((curr_alphasss*S_u)./Rate));
    if curr_lat == Inf
        quit()
        %d = 1;
    end
    trans_latencies = [curr_lat trans_latencies];
% %     max(beta2*((alphasss*S_u)./(B*log2(1+((p*ada_1*alphasss.^2.*all_states.^2)./(1+per_user_interference))))))
    
 
end
trans_latency = mean(trans_latencies);
latency = comp_latency + trans_latency;
 




% latency = max((beta1*(sum(((alphasss.*S_u*C)/f),2)))) + mean(max(beta2*((alphasss*S_u)./(B*log2(1+((p*ada_1*alphasss.^2.*all_states.^2)./(1+per_user_interference))))))); 

end

