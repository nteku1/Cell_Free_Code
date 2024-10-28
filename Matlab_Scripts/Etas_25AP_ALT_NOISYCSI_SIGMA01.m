clear all;


%% Define simulation setup
%% Leverages variables from Emil Björnson and Luca Sanguinetti, “Making Cell-Free Massive MIMO Competitive With MMSE Processing and Centralized Implementation,” IEEE Transactions on Wireless Communications, 
%% vol. 19, no. 1, pp. 77-90, January 2020.
%processing capabilities
B = 20*10^6; %bandwidth
f =  1*10^6;
C = 20;
%%%D = 10*10^6;
S_u= 1*10^3;
debug_1 = zeros(1,1000);
debug_2 = zeros(1,1000);
%Number of Monte Carlo setups
nbrOfSetups = 1;%3 30;
aaaaaa = 0;
outputs = zeros(1,1000);
const_outputs = zeros(1,1000);
best_outputs = zeros(1,1000);
indices = zeros(1,nbrOfSetups);
%Number of channel realizations per setup
nbrOfRealizations = 1; %1000;
%Number of APs in the cell-free network
L = 25;%10;%25;%25;%20;%10;%25;%20; %25; %20;% 10; %original: 10

%Number of UEs
K = 6;%6;

%Number of antennas per AP
N = 1;

%Length of the coherence block
tau_c = 200;

%Number of pilots per coherence block
tau_p = 20;

%Uplink transmit power per UE (mW)
p = 100;


% fileidfun = fopen('preapred_ver100000_finallast_last_1_29_25_6_H_ESTIMATE_MMSEE_50000_part2_ver62.txt','r');
fileidfun = fopen('prepared_ver100000_finallast_last_1_29_25_6_H_SIGMAPOINT01_NOISY_ESTIMATE_MMSEE_part2_ver62.txt','r');
fileidfun_ACTUAL = fopen('prepared_ver100000_finallast_last_1_29_25_6_H_ACTUAL_NOT_ESTIMATE_MMSEE_part2_ver62.txt','r');

id_count = 0;
INDEX_COUNTER = 0;
all_states = [];
all_gain = [];
all_states_ACTUAL = [];
all_gain_ACTUAL = [];
%log2(1+((p*ada_1*alphasss^2*abs(H_AP).^2)/(1+p*ada_1*((1-alphasss)^2)*abs(H_AP)^2)))
for iiii = 1:6000%20000%6000%4000%300000%1000
    d = str2num(fgetl(fileidfun));
    d_ACTUAL = str2num(fgetl(fileidfun_ACTUAL));
    for slen = 1:2:(L*2)-1%19
        all_gain = [all_gain abs(d(slen)+i*d(slen+1))];  
        all_gain_ACTUAL = [all_gain_ACTUAL abs(d_ACTUAL(slen)+i*d_ACTUAL(slen+1))]; 
    end
    
    all_states = [all_states all_gain.'];
    all_states_ACTUAL = [all_states_ACTUAL all_gain_ACTUAL.'];
    all_gain = [];
    all_gain_ACTUAL = [];
    id_count = id_count + 1;
    %if mod(id_count,20) ~=0
    if mod(id_count,6) ~=0
        continue
    end

    for n = 1:nbrOfSetups

        if n == 1
        alphasss_0= rand(2*L,K);
           for jajj = 1:K
             %normalize each column partway - first half of column normalized represents alphas
             alphasss_0(1:L,jajj) = alphasss_0(1:L,jajj)/sum(alphasss_0(1:L,jajj));
             %second half of column normalized represents etas
             alphasss_0(L+1:2*L,jajj) = alphasss_0(L+1:2*L,jajj)/sum(alphasss_0(L+1:2*L,jajj));
           end
        end
        % % % % else
        % % % % alphasss_0 = alphasss;
        % % % % end

        %declaring parameters for fmincon
        A = [];
        b = [];
        lb = zeros(1,2*L*K)+ 0.001 * ones(1,2*L*K);%0;
        ub = ones(1,2*L*K);
        %%% Based on https://www.mathworks.com/matlabcentral/answers/454349-multiple-linear-equality-constraints-in-fmincon
                Aeq = [[ones(1,L) zeros(1,2*K*L-L)];[zeros(1,L) ones(1,L) zeros(1,2*K*L-2*L)]; [zeros(1,2*L) ones(1,L) zeros(1,2*K*L-3*L)]; [zeros(1,3*L) ones(1,L) zeros(1,2*K*L-4*L)];  [zeros(1,4*L) ones(1,L) zeros(1,2*K*L-5*L)];,...
                    [zeros(1,5*L) ones(1,L) zeros(1,2*K*L-6*L)]; [zeros(1,6*L) ones(1,L) zeros(1,2*K*L-7*L)]; [zeros(1,7*L) ones(1,L) zeros(1,2*K*L-8*L)]; [zeros(1,8*L) ones(1,L) zeros(1,2*K*L-9*L)]; [zeros(1,9*L) ones(1,L) zeros(1,2*K*L-10*L)]; [zeros(1,10*L) ones(1,L) zeros(1,2*K*L-11*L)]; [zeros(1,2*K*L-L) ones(1,L)]];
        beq = ones(2*K,1);


     options = optimoptions(@fmincon,'Algorithm','sqp','MaxFunEvals',4500,'StepTolerance',1e-10,'TolCon',1e-10); 
     %options = optimoptions(@fmincon,'Algorithm','sqp','MaxFunEvals',4500,'StepTolerance',1e-10,'TolCon',1e-10,'TolFun',1e-12);%'MaxIter',2000,'TolCon',1e-10); %2000 TolCon 1e-10 TolX
 
      total_transmission = alphasss_0(L+1:2*L,:) .* all_states.^2;
      total_interference = sum(sum(total_transmission));
      per_user_interference = total_interference - total_transmission;
      per_user_interference = p*per_user_interference;
      
      
      alphasss = fmincon(@(alphasss)fun_ALT(alphasss,per_user_interference,all_states,L),alphasss_0,A,b,Aeq,beq,lb,ub,[],options);
 
 

        d= 1;
        
       


    end
%      outputs(iiii/K) =  fun(alphasss);
     %Debugging
     %iiii alphasss(L+1:2*L,:).
     total_transmission =  alphasss(L+1:2*L,:).* all_states_ACTUAL.^2;
      total_interference = sum(sum(total_transmission));
      per_user_interference = total_interference - total_transmission;
      per_user_interference = p*per_user_interference;
      
      

    outputs(iiii/K) =  fun_ALT(alphasss,per_user_interference,all_states_ACTUAL,L);
     

     all_states = [];
     all_states_ACTUAL = [];
     id_count = 0;
     


    
    
end


%for consistency with other methods
outputs(1) = [];
const_outputs(1) = [];
best_outputs(1) = [];


%CDF calculation 
%statsss = 10:10:300; %% chan 1
%%%%statsss = 10:50:1000;  %%% chan 2/3
%%%%statsss = 5:5:1000;  %%% chan 2/3
statsss = 0:0.001:0.2;
cdffff = zeros(1,length(statsss));
for j = 1:length(statsss)
    tata = length(find(outputs<=statsss(j)));
    cdffff(j) = (tata/999);   
end

plot(statsss, cdffff,'-r');

xlabel('Max Latency (s)');
ylabel('CDF');
