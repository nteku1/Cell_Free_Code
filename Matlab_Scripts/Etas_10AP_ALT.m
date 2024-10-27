clear all;
rng(15,"twister") 

%% Define simulation setup
%processing capabilities
B = 20*10^6; %bandwidth
f =  1*10^6;
C = 20;

S_u= 1*10^3;
nbrOfSetups = 1;
outputs = zeros(1,1000);


nbrOfRealizations = 1; %1000;

%Number of APs in the cell-free network
L = 10;

%Number of UEs
K = 6;


%Uplink transmit power per UE (mW)
p = 100;

%power ratio
ada_1 = 1;


% fileidfun = fopen('ggggetready_faaaaa1_29_50APs_singlepathloss_p2.txt','r');
% fileidfun = fopen('prepared_ver100000_finallast_last_1_29_25_6_H_ACTUAL_NOT_ESTIMATE_MMSEE_part2_ver62.txt','r');
% fileidfun = fopen('best_vvvver1_endfortoday__1_29_20APs_Single_Path_loss_p2.txt','r');
fileidfun = fopen('ACTUAL_cbest_vvvver1_endfortoday__1_29_10APs_Single_Path_loss_p2.txt','r');


id_count = 0;
all_states = [];
all_gain = [];
for iiii = 1:6000
    %reading channel coeffiencts from a block
    d = str2num(fgetl(fileidfun));
    
    for slen = 1:2:(L*2)-1%19
        all_gain = [all_gain abs(d(slen)+i*d(slen+1))];        
    end
    
    all_states = [all_states all_gain.'];
    all_gain = [];
    id_count = id_count + 1;
    if mod(id_count,6) ~=0
        continue
    end

     %for each block pass through fmnicon 4 times
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
        % % % % else %use optimized alphas/etas from last step
        % % % % alphasss_0 = alphasss;
        % % % % end

         %declaring parameters for fmincon
        A = [];
        b = [];
        lb = zeros(1,2*L*K)+ 0.001 * ones(1,2*L*K);%0;
        ub = ones(1,2*L*K);
        % Aeq = [[ones(1,L) zeros(1,K*L-L)];[zeros(1,L) ones(1,L) zeros(1,K*L-2*L)]; [zeros(1,2*L) ones(1,L) zeros(1,K*L-3*L)]; [zeros(1,3*L) ones(1,L) zeros(1,K*L-4*L)];  [zeros(1,4*L) ones(1,L) zeros(1,K*L-5*L)];[zeros(1,K*L-L) ones(1,L)]];
        %%% Based on https://www.mathworks.com/matlabcentral/answers/454349-multiple-linear-equality-constraints-in-fmincon
                Aeq = [[ones(1,L) zeros(1,2*K*L-L)];[zeros(1,L) ones(1,L) zeros(1,2*K*L-2*L)]; [zeros(1,2*L) ones(1,L) zeros(1,2*K*L-3*L)]; [zeros(1,3*L) ones(1,L) zeros(1,2*K*L-4*L)];  [zeros(1,4*L) ones(1,L) zeros(1,2*K*L-5*L)];,...
                    [zeros(1,5*L) ones(1,L) zeros(1,2*K*L-6*L)]; [zeros(1,6*L) ones(1,L) zeros(1,2*K*L-7*L)]; [zeros(1,7*L) ones(1,L) zeros(1,2*K*L-8*L)]; [zeros(1,8*L) ones(1,L) zeros(1,2*K*L-9*L)]; [zeros(1,9*L) ones(1,L) zeros(1,2*K*L-10*L)]; [zeros(1,10*L) ones(1,L) zeros(1,2*K*L-11*L)]; [zeros(1,2*K*L-L) ones(1,L)]];
        beq = ones(2*K,1);



        %declaring addditional options for fmincon, using SQP algorithm
       options = optimoptions(@fmincon,'Algorithm','sqp','MaxFunEvals',4500,'StepTolerance',1e-10,'TolCon',1e-10); 
       % options = optimoptions(@fmincon,'Algorithm','sqp','MaxFunEvals',4500,'StepTolerance',1e-10,'TolCon',1e-10,'TolFun',1e-12);

       %pre-calculations to make passing into fmincon easier
      % % total_transmission = alphasss_0.^2 .* all_states.^2;
      % % total_interference = sum(sum(total_transmission));
      % % per_user_interference = total_interference - total_transmission;
      % % per_user_interference = p*ada_1*per_user_interference;

      total_transmission = alphasss_0(L+1:2*L,:) .* all_states.^2;
      total_interference = sum(sum(total_transmission));
      per_user_interference = total_interference - total_transmission;
      per_user_interference = p*per_user_interference;
      
      %referencing external cost function fun to calculate latency, default
       alphasss = fmincon(@(alphasss)fun_ALT(alphasss,per_user_interference,all_states,L),alphasss_0,A,b,Aeq,beq,lb,ub,[],options);
 


        
       


    end
    %Debugging
    %iiii
      %post-processing calculating latency using final determined alphas from
      %fmincon
      % total_transmission = alphasss.^2 .* all_states.^2;
      total_transmission = alphasss(L+1:2*L,:).* all_states.^2;
      total_interference = sum( sum(total_transmission));
      per_user_interference = total_interference - total_transmission;
      outputs(iiii/K) =  fun_ALT(alphasss,per_user_interference,all_states,L);

     all_states = [];
     id_count = 0;

    
    
end


%for consistency with other methods, ignore results from first block
outputs(1) = [];


%CDF calculation 
statsss = 0:0.001:0.2;
cdffff = zeros(1,length(statsss));
for j = 1:length(statsss)
    tata = length(find(outputs<=statsss(j)));
    cdffff(j) = (tata/999);   
end

plot(statsss, cdffff,'-r');


xlabel('Max Latency (s)');
ylabel('CDF');

%Once this result is saved, run writing_interior_JOURNAL.m to save the CDF.
%

