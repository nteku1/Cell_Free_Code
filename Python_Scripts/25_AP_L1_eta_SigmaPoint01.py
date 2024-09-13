
#importing necessary packages
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import copy

#setting seed for reproducibility
tf.keras.utils.set_random_seed(15)

#loading in file for training data
file1 = open('prepared_ver100000_finallast_last_1_29_25_6_H_SIGMAPOINT01_NOISY_ESTIMATE_MMSEE_50000_ver62.txt','r')
lines = file1.readlines()


#setting neccessary parameters and iterating variables
num_APs = 25
num_user = 6
num_exclude = 22 #how many of the alphas to ignore. e.g. for 10 APs and num_exlcude = 7, will keep top 3 highest alphas
pos = 0
raw_iq = []
complex_vec = []
complex_vec_total = []
count = 0
count2 = 0

#reading in first set of channel coefficents for 6 UEs
for i in range(num_user):
  temp = lines[pos+i]
  temp = temp.strip()
  temp = temp.split(' ')
  raw_iq.append(temp)
  print(raw_iq[i])

for i in range(num_user):
  for jslack in raw_iq[i]:
    if count % 2 == 0:
      tens = float(raw_iq[i][count])
    if  count % 2 == 1:
      complex_vec.append(complex(tens,float(raw_iq[i][count])))
      count2 = count2+ 1
    count = count + 1
  complex_vec_total.append(complex_vec)
  complex_vec = []
  count = 0
  count2 = 0


# converting above file reading code into function
def converttt(lines,id):  
  raw_iq = []
  complex_vec = []
  complex_vec_total = []
  count = 0
  count2 = 0

  for i in range(num_user):
    temp = lines[id+i]
    temp = temp.strip()
    temp = temp.split(' ')
    raw_iq.append(temp)

  for i in range(num_user):
    for jslack in raw_iq[i]:
      if count % 2 == 0:
        tens = float(raw_iq[i][count])
      if  count % 2 == 1:
        complex_vec.append(complex(tens,float(raw_iq[i][count])))
        count2 = count2+ 1
      count = count + 1
    complex_vec_total.append(complex_vec)
    complex_vec = []
    count = 0
    count2 = 0

  return complex_vec_total

#setting more parameters for UEs
freq = 1*10**6
B = 20*10**6 # Bandwidth
C = 20
S_u = 10**3 # data size
p = 100 #100 #(mW) as in [7] in paper
ada_1 = 1 # power control factor
num_actions = num_APs
num_states = 2*num_APs # input to neural network requires alphas of each AP and channel gains between each AP and the user


# function for using neural network to generate alphas. takes in channel gains and snrs of all 6 UEs at current timestep
def update(
        state_batch, gain_critic
):
    with tf.GradientTape() as tape:
        #variable initialization
        comp_lats = []
        trans_lats = []
        critic_value = tf.zeros([num_APs],dtype='float64')
        critic_value_etas = tf.zeros([num_APs],dtype='float64')
        oldies = tf.zeros([num_APs], dtype=tf.dtypes.float64)
        rat = tf.zeros([num_APs], dtype=tf.dtypes.float64)

        anti_indices = []
        anti_update = []
        copy_crit_indices = []
        state_idx = 0

        for jslack in gain_critic:
            #reading in gain for each UE and combining it with with uplink SNRs of that UE
            UE_state = np.concatenate((np.array(state_batch[state_idx]),np.array(jslack)))
            #passing state into neural network
            temp_fw = cell_nn((tf.expand_dims(UE_state, 0)),training=True)[0]
            temp_fw = tf.cast(temp_fw, dtype='float64')
            #First half of output are message allocation (alphas). Second half are power factors (etas).
            temp_alphas = tf.slice(temp_fw, [0],[num_APs])
            temp_etas = tf.slice(temp_fw,[num_APs],[num_APs])
            #Sorting highest alphas and subsequently etas.
            temp_indices = np.argsort(temp_alphas)[num_exclude:]
            temp_count = 0


            for isl in range(len(temp_alphas)):
                # if the current alpha/eta is not one of the highest set to zero
                if temp_count not in temp_indices:
                    temp_alphas = tf.tensor_scatter_nd_update(temp_alphas, [[temp_count]], [0])
                    temp_etas = tf.tensor_scatter_nd_update(temp_etas, [[temp_count]], [0])
                temp_count += 1

            #keeping track of indicies of non-tau highest values.

            crit_indices = tf.experimental.numpy.where(temp_alphas == 0)
            copy_crit_indices.append(crit_indices[0])
            crit_list = []
            cn = 0

            for ist in range(len(crit_indices[0])):
                crit_list.append([])
                for istt in range(1):
                    crit_list[ist].append(crit_indices[0][istt + cn])
                cn += 1

            anti_indices.append(crit_list)
            update_list = []

            for upp in range(len(crit_indices[0])):
                update_list.append(0)
            anti_update.append(update_list)


            #Normalizing top highest alphas and etas
            temp_alphas = temp_alphas / tf.math.reduce_sum(temp_alphas)
            temp_etas = temp_etas / tf.math.reduce_sum(temp_etas)
            critic_value = tf.concat([critic_value, temp_alphas], axis=0)
            critic_value_etas = tf.concat([critic_value_etas, temp_etas], axis=0)

            state_idx += 1

        critic_value = critic_value[num_APs:]
        critic_value_etas = critic_value_etas[num_APs:]
        #iterating over UEs
        for gan in range(num_user):
            chansd_other = tf.zeros([num_APs], dtype=tf.dtypes.float64)
            #grabbing etas of current UE
            eta_train = critic_value_etas[gan * num_APs:num_APs + gan * num_APs]
            #grabing etas of all other UEs
            tf_other_etas = tf.concat([critic_value_etas[:gan*num_APs],critic_value_etas[num_APs+gan*num_APs:]],0)
            tf_other_etas = tf.reshape(tf_other_etas,[num_user-1,num_APs])


            gain_vec = gain_critic[gan]
            #applying zeros to channel gains  of current UE whose split splits to an AP were set to zero
            if len(anti_indices[gan]) != 0 and len(anti_update[gan]) != 0:
                gain_vec = tf.tensor_scatter_nd_update(gain_vec, anti_indices[gan], anti_update[gan])
            chansd = ((gain_vec) ** 2)

            #grabbing channel gains of all other UEs
            for extra_gan in range(num_user):
                #skipping current UE
                if extra_gan == gan:
                    continue
               #applying zeros to channel gains of all other UEs whose splits to an AP were set to zero
                elif len(anti_indices[extra_gan]) != 0 and len(anti_update[extra_gan]) != 0:
                    chansd_other = tf.concat([chansd_other, tf.tensor_scatter_nd_update(gain_critic[extra_gan], anti_indices[extra_gan], anti_update[extra_gan])],axis=0)

            #calculating interference caused by all other UEs' transmissions
            chansd_other = chansd_other[num_APs:]
            chansd_other = tf.reshape(chansd_other,[num_user-1,num_APs])
            chansd_other = tf.multiply(chansd_other,chansd_other)
            tens_other_interference = tf.multiply(tf_other_etas,chansd_other)
            tens_other_interference = tf.math.reduce_sum(tens_other_interference)
            tens_temp_interference = tf.multiply(eta_train, chansd)


            #calculating vector of SINRs for current UE.
            dpp = (p *  (tf.math.reduce_sum(tens_temp_interference) - tens_temp_interference)) + ((p)*tens_other_interference)
            #dpp = ((p) *  tf.math.reduce_sum(tens_temp_interference) - tens_temp_interference) + ((p)*tens_other_interference)
            tep = (((p) * (eta_train * chansd)))


            #calculating rates
            rate_constant = tf.constant(1, dtype='float64')
            oldies = tf.concat([oldies,((tep) / (rate_constant + dpp))],axis=0)
            temp_rat= ((B * np.log2((rate_constant + ((tep) / (1 + dpp))))))
            rat = tf.concat([rat, temp_rat], axis=0)


        #skipping first row of zeros that came with initialization
        rat = rat[num_APs:]
        oldies = oldies[num_APs:]

       #iterating over users
        for vin in range(num_user):
            #zeroing out expressions for alphas of UE that were set to zero
            if len(anti_indices[vin]) != 0 and len(anti_update[vin]) != 0:
                new_rat = tf.tensor_scatter_nd_update(rat[vin * num_APs:num_APs + vin * num_APs], anti_indices[vin], anti_update[vin])
            else:
                new_rat = rat
            new_rat = new_rat[new_rat != 0]
            #Appending computational latencies due to each UE
            comp_lats.append(critic_value[vin * num_APs:num_APs + vin * num_APs] * S_u)
            #Appending largest transmission latency of each UE
            trans_lats.append(tf.math.reduce_max((
            critic_value[vin * num_APs:num_APs + vin * num_APs][
                critic_value[vin * num_APs:num_APs + vin * num_APs] != 0] * S_u / new_rat)))

        #Calculating computational latencies across APs.
        tf_actual_comp = tf.math.reduce_sum(comp_lats, axis=0)
        #Storing largest computational latency
        tf_actual_comp = tf.math.reduce_max((tf_actual_comp * C) / freq)
        #Storing average of largest transmission latency
        tf_actual_trans = tf.math.reduce_mean(trans_lats)
        #Calculating total latency
        final_lats = tf_actual_comp + tf_actual_trans


        #Setting Loss as attained latency
        critic_loss = final_lats

    #Applying ADAM optimizer
    critic_grad = tape.gradient(critic_loss, cell_nn.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_grad, cell_nn.trainable_variables)
    )

    return critic_value,critic_value_etas,critic_loss,oldies

#function for creating neural network
def nn_init():
    state_input = layers.Input(shape=(num_states)) # 16 128 24000 - could increase more  or try 6000        128,64 12000??

    out = layers.Dense(128, activation="mish", activity_regularizer='l2')(state_input)  #128 mish  #128 mish        #128 mish    #64 relu
    out = layers.Dense(64, activation="relu", activity_regularizer='l2')(out)           #64 tanh   #64  relu        #64 swish   #128 selu



    #out = layers.Dense(64, activation="mish", activity_regularizer='l2')(state_input) #16 swish
    #out = layers.Dense(128, activation="swish", activity_regularizer='l2')(out) #32gelu

    #out = layers.Dense(16, activation="relu", activity_regularizer='l2')(state_input) #16 swish
    #out = layers.Dense(16, activation="selu", activity_regularizer='l2')(out) #32gelu        128 swish
    #out = layers.Dense(16, activation="tanh", activity_regularizer='l2')(out)

    #out = layers.Dense(32,64, activation="mish", activity_regularizer='l2')(state_input) #16 swish
    #out = layers.Dense(128, activation="swish", activity_regularizer='l2')(out) #32gelu        128 swish


    #out = layers.Dense(16, activation="swish", activity_regularizer='l2')(state_input) #16 swish
    #out = layers.Dense(128, activation="swish", activity_regularizer='l2')(out) #32gelu        128 swish
    #out = layers.Dense(16, activation="swish", activity_regularizer='l2')(out) #16 tanh              16 swish
    outputs = layers.Dense(num_states,activation="sigmoid")(out)
    model = tf.keras.Model(state_input, outputs)

    return model


#function for generating neural networks during testing
def policy_testing(state):
    #Get alphas from raw neural network oput
    sampled_actions = tf.squeeze(cell_nn(state,training=False))

    legal_action = sampled_actions.numpy()
    legal_alphas = legal_action[0:num_APs]
    legal_etas = legal_action[num_APs:num_states]
    #Get top tau highest alphas/etas
    legal_indices = np.argsort(legal_alphas)[num_exclude:]
    id_count = 0
    #setting non top highest alphas/etas to zeros
    for i in range(len(legal_alphas)):
        if id_count not in legal_indices:
            legal_alphas[id_count] = 0
            legal_etas[id_count] = 0
        id_count += 1

     
    #normalize alphas/etas
    legal_alphas = legal_alphas / np.sum(legal_alphas)
    legal_etas = legal_etas / np.sum(legal_etas)

    legal_final_action = np.concatenate((legal_alphas,legal_etas))

    return [np.squeeze(legal_final_action)]


#initialziing neural network prameters
cell_nn = nn_init()

cell_nn_lr = 0.0002 #learning rate

critic_optimizer = tf.keras.optimizers.Adam(cell_nn_lr) #adam

total_episodes = 1

#tracking training performance of neural network
ep_reward_list = []

indexxx = 0
stop_now = 0


#Strat training phase
for ep in range(total_episodes):

    prev_state = np.zeros([num_states,])
    episodic_reward = 0
    alphasss = []

    if stop_now == 1:
      break
    


    #declaring random alphas/etas for each UE at initial timestep
    for i in range(num_user):
      rand_vecs = np.random.random(size=(1,num_states))
      rand_vecs[0][0:num_APs] = rand_vecs[0][0:num_APs] /np.sum(rand_vecs[0][0:num_APs])
      rand_vecs[0][num_APs:num_states] = rand_vecs[0][num_APs:num_states] / np.sum(rand_vecs[0][num_APs:num_states])
      alphasss.append(rand_vecs[0])
    


    indexxx = ep*200

    while indexxx < 200 + 200*500:
        if indexxx == 0:
          actions = []
          #Appending alphas for each user
          for ikea in range(num_user):
            actions.append(np.asarray(alphasss[ikea]))

        #Variable initialization
        comp_latencies_co = []
        trans_latencies_co = []
        latencies_co = []
        copy_of_latencies_co = []
        copy_of_products_co = []
        copy_of_SINRS_co = []
        sum_temp = np.zeros(shape=(1,num_APs))
        sum_den = np.zeros(shape=(1,num_APs))
        un_dB_sinr = []
        #Iterating over each UE
        for swag in range(num_user):
            action = actions[swag]

            alphasss = action[0:num_APs]
            etas = action[num_APs:num_states]
            other_etas = []
            other_gains = []
            #getting channel gains of all other UEs
            chan_others_squared = (np.absolute(complex_vec_total[:swag] + complex_vec_total[swag+1:]) ** 2)
            #getting etas of all other UEs
            for iterating in range(num_user):
                if iterating == swag:
                    continue
                else:
                    other_etas.append(actions[iterating][num_APs:num_states])
            other_etas = np.asarray(other_etas)
            #calculating interference due to other UEs
            other_interference = np.multiply(other_etas, chan_others_squared)
            other_interfernece = np.sum(other_interference)
            #calculating power of current UE
            chan_squared = (np.absolute(complex_vec_total[swag]) ** 2)
            user_interference = []
            #Calculating SINR
            temp_interference = np.multiply(etas, chan_squared)
            for j in range(num_APs):
                user_interference.append(np.sum(temp_interference,axis=0) - temp_interference[j])

            user_interference = np.asarray(user_interference)
            den = ((p)*user_interference) + (p)*other_interfernece
            temp = (((p) * (etas * chan_squared)))
            #storing SINR (this only applies for first timestep)
            un_dB_sinr.append(temp/(1+den))


        print('step')
        print(indexxx)
        if indexxx == 0: #first time step keep above sinrs.
            debug = 1
        else:
            un_dB_sinr = []
            un_dB_sinr = (type_cast_sinr)  #for subsequent time step keep sinrs of previous timestep.



        indexxx = indexxx + num_user
        if indexxx < len(lines):
            #get gains from training set
            complex_vec_total = (converttt(lines, indexxx))
            copy_of_GAINS_co = np.absolute(complex_vec_total)

        #pass SINRs from previous timestep and gains from current timestep into neural network
        resulting_alphas,resulting_etas,nn_latency,old_sinr = update(un_dB_sinr, copy_of_GAINS_co)
        actions = []
        type_cast_sinr = []
        #collect resulting alphas/etas and sinrs.
        for gan_step in range(num_user):
          xtemp = np.concatenate((resulting_alphas[gan_step * num_APs:num_APs + gan_step * num_APs].numpy(),resulting_etas[gan_step * num_APs:num_APs + gan_step * num_APs].numpy()))
          actions.append(xtemp)
          #Debugging printing alphas
          #print('Alphasss')
          #print(actions[gan_step])

        for AP_id in range (num_user):
            type_cast_sinr.append(old_sinr[AP_id * num_APs:num_APs + AP_id * num_APs].numpy())



        #Once training process reaches set timestep, stop training phase
        if indexxx == 78000:#36000:#12000:#36000:#24000:#24000:#87000:#72000:
            stop_now = 1
            break

        #For debugging, printing rewards during training process
        print('Reward')
        print(nn_latency.numpy())

        



        



#printing training index
print(indexxx)


# Save the weights
cell_nn.save_weights("sam260.h5")

# opening file with testing set containing noisy CSI
file2 = open('prepared_ver100000_finallast_last_1_29_25_6_H_SIGMAPOINT01_NOISY_ESTIMATE_MMSEE_part2_ver62.txt','r')
testlines = file2.readlines()


#opening file with test set containing perfect CSI
file3 = open('prepared_ver100000_finallast_last_1_29_25_6_H_ACTUAL_NOT_ESTIMATE_MMSEE_part2_ver62.txt','r')
testlines_ACTUAL_H = file3.readlines()


#resetting file iterating parameters
pos = 0
raw_iq = []
raw_iq_ACT = []
complex_vec2 = []
complex_vec_total_p2 = []
count = 0
count2 = 0
complex_vec2_ACT = []
complex_vec_total_p2_ACT = []

#again reading opening for test set
for i in range(num_user):
  temp = testlines[pos+i]
  temp = temp.strip()
  temp = temp.split(' ')
  raw_iq.append(temp)
  print(raw_iq[i])

  temp_ACT = testlines_ACTUAL_H[pos+i]
  temp_ACT = temp_ACT.strip()
  temp_ACT = temp_ACT.split(' ')
  raw_iq_ACT.append(temp_ACT)

for i in range(num_user):
  for jslack in raw_iq[i]:
    if count % 2 == 0:
      tens = float(raw_iq[i][count])
      tens_ACT = float(raw_iq_ACT[i][count])
    if  count % 2 == 1:
      complex_vec2.append(complex(tens,float(raw_iq[i][count])))
      complex_vec2_ACT.append(complex(tens_ACT, float(raw_iq_ACT[i][count])))
      count2 = count2+ 1
    count = count + 1
  complex_vec_total_p2.append(complex_vec2)
  complex_vec_total_p2_ACT.append(complex_vec2_ACT)
  complex_vec2 = []
  complex_vec2_ACT = []
  count = 0
  count2 = 0



#load weights for testing
cell_nn.load_weights("sam260.h5")


# Declaring result-collecting variables
ep_reward_list = []
const_reward_list = []
unif_reward_list = []
indexxx = 0
total_episodes = 1

#Beginning testing phase
for ep in range(total_episodes):

    #variable initialization
    prev_state = np.zeros([num_states,])
    episodic_reward = 0
    alphasss = []

    #For first timestep need to generate random alphas/etas and normalize
    for i in range(num_user):
      rand_vecs = np.random.random(size=(1,num_states))
      rand_vecs[0][0:num_APs] = rand_vecs[0][0:num_APs] /np.sum(rand_vecs[0][0:num_APs])
      rand_vecs[0][num_APs:num_states] = rand_vecs[0][num_APs:num_states] / np.sum(rand_vecs[0][num_APs:num_states])
      alphasss.append(rand_vecs[0])


    while indexxx < len(testlines)-(num_user-1):
        #clearing input to neural network for processing
        tf_prev_state = [[]]*num_user
        if indexxx > 0:
            #if not at initial timestep, record the previous SINRs
          for jwiw in range(num_user):
            tf_prev_state[jwiw] = (tf.expand_dims(tf.convert_to_tensor(prev_state[jwiw],dtype=np.float32), 0))
        else:
          for jwiw in range(num_user):
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state,dtype=np.float32), 0)

       #clearing alphas/etas for processing
        actions = []
        
        if indexxx == 0:
          #for first timestep using random alphas
          for ikea in range(num_user):
            actions.append(np.asarray(alphasss[ikea]))
        else:
          #for all subsequent timesteps use alphas/etas generated by neural network
          for ikea22 in range(num_user):
            actions.append(policy_testing(tf_prev_state[ikea22])[0])

        #variable initialization for neural network, uniform, and greedy methods.
        comp_latencies_co = []
        trans_latencies_co = []
        comp_latencies_u_co = []
        trans_latencies_u_co = []
        comp_latencies_b_co = np.zeros(shape=(1,num_APs))
        trans_latencies_b_co = []
        latencies_co = []
        latencies_u_co = []
        latencies_b_co = []
        sum_temp = np.zeros(shape=(1,num_APs))
        sum_den = np.zeros(shape=(1,num_APs))
        sum_temp_u = np.zeros(shape=(1,num_APs))
        sum_den_u = np.zeros(shape=(1,num_APs))
        un_dB_sinr = []
        un_dB_sinr_ACT = []
        rate = []
        rate_ACT = []

        for swag in range(num_user):
          action = actions[swag]
          other_etas = []
          #Same SINR calculation as above in training phase for each UE
          alphasss = action[0:num_APs]
          etas = action[num_APs:num_states]
          chan_others_squared = (np.absolute(complex_vec_total_p2[:swag] + complex_vec_total_p2[swag + 1:]) ** 2)
          chan_others_squared_ACT = (np.absolute(complex_vec_total_p2_ACT[:swag] + complex_vec_total_p2_ACT[swag + 1:]) ** 2)
          for iterating in range(num_user):
              if iterating == swag:
                  continue
              else:
                  other_etas.append(actions[iterating][num_APs:num_states])
          other_etas = np.asarray(other_etas)
          other_interference = np.multiply(other_etas, chan_others_squared)
          other_interfernece = np.sum(other_interference)
          other_interference_ACT = np.multiply(other_etas, chan_others_squared_ACT)
          other_interfernece_ACT = np.sum(other_interference_ACT)
          user_interference = []
          user_interference_ACT = []
          chan_squared = (np.absolute(complex_vec_total_p2[swag])**2)
          chan_squared_ACT = (np.absolute(complex_vec_total_p2_ACT[swag]) ** 2)
          temp_interference = np.multiply(etas, chan_squared)
          temp_interference_ACT = np.multiply(etas, chan_squared_ACT)
          for j in range(num_APs):
              user_interference.append(np.sum(temp_interference, axis=0) - temp_interference[j])
              user_interference_ACT.append(np.sum(temp_interference_ACT, axis=0) - temp_interference_ACT[j])


          user_interference = np.asarray(user_interference)
          user_interference_ACT = np.asarray(user_interference_ACT)
          den = ((p) * user_interference) + ((p) * other_interfernece)
          den_ACT = ((p) * user_interference_ACT) + ((p) * other_interfernece_ACT)
          temp = (((p) * (etas * chan_squared)))
          temp_ACT = (((p) * (etas * chan_squared_ACT)))
          un_dB_sinr.append((temp) / (1 + den))
          rate.append(((B * np.log2((1 + ((temp) / (1 + den)))))))
          rate_ACT.append(((B * np.log2((1 + ((temp_ACT) / (1 + den_ACT)))))))


        for sew in range(num_user):
          #checking for where alphas are zero for each UE
          curr_alphas = actions[sew][0:num_APs]
          alphasss_indices = np.where(curr_alphas==0)[0]

          print('Alphas Indices')
          print(alphasss_indices)


          new_rate = copy.deepcopy(rate[sew])
          new_rate_ACT = copy.deepcopy(rate_ACT[sew])

          # setting rates where alpha = 0 to zero to avoid Inf or NaNs
          for i in alphasss_indices:
            new_rate[i] = 0
            new_rate_ACT[i] = 0

          print('Alphasss')
          print(curr_alphas)

          new_alphasss = curr_alphas[curr_alphas != 0]
          
          observing_rate_indices = np.where(new_rate==0)[0]
          observing_rate_indices_ACT = np.where(new_rate_ACT == 0)[0]
          observing_alphasss_indices = np.where(curr_alphas== 0)[0]

          tezerk = len(observing_rate_indices_ACT)
          tezerk2 = len(observing_alphasss_indices)

          if tezerk != tezerk2:
            print('Size issue')
            print('chan squared')
            print(chan_squared)
            print('Alphas')
            print(alphasss)
            print('Temp')
            print(temp)
            print('Den')
            print(den)
            print('Rate')
            print(rate[0])
            print('New Rate')
            print(new_rate)
            quit()

          for excak in range(tezerk):
            if observing_rate_indices[excak] != observing_alphasss_indices[excak]:
              print('issue')
              print(chan_squared)
              print('Alphas')
              print(alphasss)
              print('Temp')
              print(temp)
              print('Den')
              print(den)
              print('Rate')
              print(rate[0])
              print('New Rate')
              print(new_rate)
              quit()
          



          #Perform rate calculations on non-zero alphas/etas
          new_rate = new_rate[new_rate != 0]
          new_rate_ACT = new_rate_ACT[new_rate_ACT != 0]
          product = curr_alphas*S_u

          product = product[product!=0]

          comp_latencies_co.append(((curr_alphas * S_u)))
          trans_latencies_co.append(np.max((product/new_rate)))


        actual_comp_lat =  np.sum(comp_latencies_co,axis=0)
        actual_comp_lat = np.max((actual_comp_lat*C)/freq)
        actual_trans_lat = np.mean(trans_latencies_co)
        final_rewards = actual_comp_lat + actual_trans_lat


        reward = final_rewards

        #making sure reward collected from first time step is not counted as it it was obtained with random alphas
        if indexxx >0 and indexxx < len(testlines):
          ep_reward_list.append(reward)


        ##### UNIFORM
        actual_trans_lat_u = []
        for delicious in range(num_user):
          #Splitting up alphas/etas uniformly for each user
          unif_alphasss = 1/num_APs*np.ones([1,num_APs])
          unif_etas = 1 / num_APs * np.ones([1, num_APs])
          #Same procedure as neural network for calculating SINR
          product_u = unif_alphasss*S_u
          chan_squared_u = (np.absolute(complex_vec_total_p2_ACT[delicious])**2)
          chan_others_squared_u = (np.absolute(complex_vec_total_p2_ACT[:delicious] + complex_vec_total_p2_ACT[delicious + 1:]) ** 2)
          other_interference_u = np.multiply(unif_etas, chan_others_squared_u)
          other_interference_u = np.sum(other_interference_u)
          user_interference_u = []
          temp_interference_u = np.multiply(unif_etas, chan_squared_u)[0]
          for j in range(num_APs):
              user_interference_u.append(np.sum(temp_interference_u, axis=0) - temp_interference_u[j])

          user_interference_u = np.asarray(user_interference_u)
          den_u = ((p) * user_interference_u) + ((p)*other_interference_u)
          temp_u = (((p) * (unif_etas * chan_squared_u)))

          rate_u = ((B * np.log2((1 + ((temp_u) / (1 + den_u))))))
          actual_trans_lat_u.append(np.max((product_u / rate_u)))
          comp_latencies_u_co.append((unif_alphasss*S_u))



          

        actual_comp_lat_u =  np.sum(comp_latencies_u_co,axis=0)
        actual_comp_lat_u = np.max((actual_comp_lat_u*C)/freq)


        actual_trans_lat_uu = np.mean(actual_trans_lat_u)
        #calculating final latency for uniform method
        reward_u = actual_trans_lat_uu + actual_comp_lat_u
        

        if indexxx >0 and indexxx < len(testlines):
          unif_reward_list.append(reward_u)


        #### Greedy Method
        best_channel_gains = []
        final_best_reward = []
        sum_temp_b = []
        #for one chunk full power
        ada_1_g = 1

        for great in range(num_user):
            # picking AP with largest NOISY channel gain
          best_id = np.argmax(np.abs(complex_vec_total_p2)[great])
          best_channel_gains.append(best_id)
          #Giving all of current UE data to one AP
          best_alphasss = 1
          product_b = best_alphasss*S_u

          # using real channel gain to calculate latency
          chan_squared_b = (np.absolute(complex_vec_total_p2_ACT[great][best_id]) ** 2)


          temp_b = (((p) * (ada_1_g * chan_squared_b)))
          sum_temp_b.append(temp_b)


        best_sums = []
        gred_idx = 0
        for solovvv in best_channel_gains:
          #Calculating SINR/rate for each UE
          comp_latencies_b_co[0][solovvv] = comp_latencies_b_co[0][solovvv] + ((best_alphasss*S_u))


          gred_num = sum_temp_b[gred_idx]
          gred_interference = sum_temp_b[:gred_idx] + sum_temp_b[gred_idx+1:]
          gred_interference = np.sum(gred_interference)
          rate_b = ((B * np.log2((1 + ((gred_num) / (1 + gred_interference))))))
          #Calculating transmission latency for greedy method
          trans_latencies_b_co.append(((product_b / rate_b)))
          gred_idx += 1

        #Calculating total latency for greedy method
        actual_comp_lat_b = np.max((comp_latencies_b_co[0] * C) / freq)
        actual_trans_lat_b = np.mean(trans_latencies_b_co)
        final_rewards_b = actual_comp_lat_b + actual_trans_lat_b




        reward_b = (final_rewards_b)

      
       

        if indexxx > 0 and indexxx < len(testlines):
          const_reward_list.append(reward_b)

        print('Step')
        print(indexxx)
        indexxx = indexxx + num_user
        if indexxx < len(testlines):
          #getting noisy channel gains for current timestep
          complex_vec_total_p2 = converttt(testlines,indexxx)
          #getting perfect CSI for current time step
          complex_vec_total_p2_ACT = converttt(testlines_ACTUAL_H, indexxx)
          state = []
          row_id = 0
          for iii in range(num_user):
            status =  np.zeros(shape=(1,num_states))[0]
            for jwe in range(num_states):
              #appending SINRs from previous timestep to input of neural network
              if jwe < num_APs:
                status[jwe] = un_dB_sinr[row_id][jwe]
              #appending current channel gains to input of neural network
              else:
                status[jwe] = (np.absolute(complex_vec_total_p2[iii][jwe-num_APs]))
              
            #appending each input for each UE
            status= status.astype('float64')
            state.append(status)
            row_id += 1
            
           

        else:
          print('Stop')
          break
        

        #used for passing whole input to neural network
        prev_state = state





#Debugging
#print(len(testlines))
#print(len(ep_reward_list))
#print(len(unif_reward_list))
#print(len(const_reward_list))


#CDF results from centralized methods Interior-Point and Sequential Quadratic Programming generated using Matlab scripts.
###min_interior_cdf = [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00100100, 0.01401401, 0.03703704, 0.08508509, 0.13113113, 0.18018018, 0.22622623, 0.27827828, 0.33133133, 0.36936937, 0.41141141, 0.43743744, 0.46046046, 0.49249249, 0.52652653, 0.54754755, 0.56856857, 0.59459459, 0.61261261, 0.63463463, 0.65165165, 0.66566567, 0.67767768, 0.68768769, 0.70070070, 0.70870871, 0.72072072, 0.72772773, 0.74074074, 0.74574575, 0.75275275, 0.75475475, 0.76076076, 0.76376376, 0.77077077, 0.77577578, 0.78178178, 0.78678679, 0.79079079, 0.79779780, 0.80880881, 0.81181181, 0.81581582, 0.81981982, 0.82082082, 0.82382382, 0.83383383, 0.83883884, 0.84484484, 0.84784785, 0.85085085, 0.85685686, 0.86086086, 0.86686687, 0.86886887, 0.87587588, 0.87687688, 0.87687688, 0.87887888, 0.87987988, 0.88488488, 0.88688689, 0.88888889, 0.88988989, 0.89089089, 0.89389389, 0.89489489, 0.89689690, 0.90190190, 0.90390390, 0.90490490, 0.90490490, 0.90690691, 0.91091091, 0.91491491, 0.91691692, 0.91791792, 0.92092092, 0.92092092, 0.92292292, 0.92292292, 0.92492492, 0.92592593, 0.92692693, 0.92792793, 0.92792793, 0.92992993, 0.93193193, 0.93293293, 0.93593594, 0.93593594, 0.93693694, 0.93893894, 0.94194194, 0.94194194, 0.94394394, 0.94394394, 0.94494494, 0.94594595, 0.94594595, 0.94694695, 0.94794795, 0.94894895, 0.94994995, 0.94994995, 0.95095095, 0.95095095, 0.95095095, 0.95095095, 0.95295295, 0.95295295, 0.95395395, 0.95395395, 0.95395395, 0.95395395, 0.95395395, 0.95395395, 0.95495495, 0.95495495, 0.95495495, 0.95495495, 0.95695696, 0.95695696, 0.95695696, 0.95695696, 0.95795796, 0.95795796, 0.95795796, 0.95895896, 0.95895896, 0.95995996, 0.96096096, 0.96096096, 0.96096096, 0.96196196, 0.96196196, 0.96196196, 0.96196196, 0.96196196, 0.96196196, 0.96196196, 0.96296296, 0.96296296, 0.96296296, 0.96296296, 0.96296296, 0.96296296, 0.96296296, 0.96396396, 0.96396396, 0.96496496, 0.96496496, 0.96596597, 0.96596597, 0.96596597, 0.96696697, 0.96696697, 0.96696697, 0.96696697, 0.96696697, 0.96796797, 0.96796797, 0.96996997, 0.96996997, 0.96996997, 0.96996997, 0.96996997, 0.96996997, 0.96996997, 0.97097097, 0.97197197, 0.97197197, 0.97197197, 0.97197197, 0.97197197, 0.97197197, 0.97197197, 0.97197197, 0.97297297, 0.97297297, 0.97397397, 0.97397397, 0.97397397, 0.97397397, 0.97397397, 0.97497497]
#min_sqp = [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.90690691, 0.92992993, 0.94594595, 0.95795796, 0.96396396, 0.96996997, 0.97597598, 0.98098098, 0.98398398, 0.98498498, 0.98898899, 0.99099099, 0.99399399, 0.99499499, 0.99599600, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99899900, 0.99899900, 0.99899900, 0.99899900, 0.99899900, 0.99899900, 0.99899900, 0.99899900, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000]
###min_sqp = [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.90690691, 0.92992993, 0.94594595, 0.95795796, 0.96396396, 0.96996997, 0.97597598, 0.98098098, 0.98398398, 0.98498498, 0.98898899, 0.99099099, 0.99399399, 0.99499499, 0.99599600, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99799800, 0.99899900, 0.99899900, 0.99899900, 0.99899900, 0.99899900, 0.99899900, 0.99899900, 0.99899900, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000, 1.00000000]

###min_interior_cdf = [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00200200, 0.00300300, 0.00400400, 0.00500501, 0.00800801, 0.01001001, 0.01001001, 0.01101101, 0.01401401, 0.02002002, 0.02202202, 0.02402402, 0.02602603, 0.03103103, 0.03603604, 0.04004004, 0.04304304, 0.04804805, 0.05105105, 0.05305305, 0.05605606, 0.05905906, 0.06106106, 0.06306306, 0.06706707, 0.07207207, 0.07507508, 0.07707708, 0.08308308, 0.08608609, 0.08908909, 0.09009009, 0.09209209, 0.09809810, 0.10410410, 0.11211211, 0.11511512, 0.11711712, 0.11911912, 0.12112112, 0.12712713, 0.13213213, 0.13513514, 0.14114114, 0.14314314, 0.14814815, 0.15415415, 0.16016016, 0.16116116, 0.16816817, 0.17017017, 0.17617618, 0.18218218, 0.18418418, 0.18718719, 0.18818819, 0.19119119, 0.19419419, 0.19719720, 0.19919920, 0.20420420, 0.20720721, 0.21221221, 0.21521522, 0.22022022, 0.22622623, 0.22922923, 0.23723724, 0.23923924, 0.24324324, 0.24624625, 0.24724725, 0.25125125, 0.25425425, 0.26226226, 0.26826827, 0.27427427, 0.27627628, 0.27727728, 0.27927928, 0.28228228, 0.28328328, 0.28528529, 0.28928929, 0.29129129, 0.29329329, 0.29929930, 0.30230230, 0.30330330, 0.30730731, 0.30930931, 0.31231231, 0.31631632, 0.31731732, 0.31931932, 0.32132132, 0.32532533, 0.32732733, 0.33033033, 0.33233233, 0.33433433, 0.33633634, 0.33733734, 0.33933934, 0.34534535, 0.34834835, 0.34934935, 0.34934935, 0.35135135, 0.35635636, 0.35835836, 0.36236236, 0.36336336, 0.36536537, 0.37037037, 0.37137137, 0.37437437, 0.37637638, 0.37837838, 0.38138138, 0.38238238, 0.38238238, 0.38238238, 0.38238238, 0.38438438, 0.38838839, 0.39239239, 0.39339339, 0.39739740, 0.39739740, 0.39939940, 0.40140140, 0.40340340, 0.40540541, 0.40740741, 0.40840841, 0.40940941, 0.41141141, 0.41341341, 0.41641642, 0.41641642, 0.41941942, 0.41941942, 0.42542543, 0.42742743, 0.42942943, 0.43143143, 0.43343343, 0.43443443, 0.43643644, 0.43743744, 0.44144144, 0.44144144, 0.44244244, 0.44444444, 0.44644645, 0.44844845, 0.45045045, 0.45145145, 0.45245245, 0.45345345, 0.45545546, 0.45645646, 0.45845846, 0.46046046, 0.46146146, 0.46346346, 0.46446446, 0.46546547, 0.47047047, 0.47247247, 0.47347347, 0.47347347, 0.47447447, 0.47547548, 0.47647648, 0.47847848, 0.47947948, 0.48048048]
###min_sqp = [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00100100, 0.00300300, 0.00300300, 0.00500501, 0.00700701, 0.00800801, 0.00800801, 0.00900901, 0.00900901, 0.01001001, 0.01201201, 0.01301301, 0.01401401, 0.01501502, 0.02102102, 0.02102102, 0.02102102, 0.02202202, 0.02302302, 0.02502503, 0.02602603, 0.02702703, 0.03003003, 0.03403403, 0.03903904, 0.04204204, 0.04704705, 0.05305305, 0.06006006, 0.06406406, 0.06706707, 0.07107107, 0.07407407, 0.07607608, 0.08108108, 0.08508509, 0.08608609, 0.08808809, 0.09209209, 0.09309309, 0.09509510, 0.10010010, 0.10310310, 0.10710711, 0.11011011, 0.11411411, 0.11711712, 0.11711712, 0.12312312, 0.12512513, 0.13513514, 0.13913914, 0.14514515, 0.14914915, 0.15215215, 0.16216216, 0.16416416, 0.16516517, 0.17017017, 0.17517518, 0.17817818, 0.18418418, 0.19019019, 0.19419419, 0.19719720, 0.19919920, 0.20320320, 0.20520521, 0.20920921, 0.21321321, 0.21821822, 0.22122122, 0.22222222, 0.22522523, 0.22622623, 0.23023023, 0.23623624, 0.23923924, 0.24624625, 0.24824825, 0.25325325, 0.25625626, 0.25925926, 0.26026026, 0.26226226, 0.26526527, 0.26826827, 0.27127127, 0.27527528, 0.27927928, 0.28128128, 0.28528529, 0.28928929, 0.29329329, 0.29429429, 0.29729730, 0.29929930, 0.30230230, 0.30530531, 0.30630631, 0.30730731, 0.31031031, 0.31131131, 0.31231231, 0.31231231, 0.31231231, 0.31531532, 0.31931932, 0.32332332, 0.32832833, 0.33333333, 0.33633634, 0.33833834, 0.34434434, 0.34534535, 0.34734735, 0.35035035, 0.35335335, 0.35435435, 0.35835836, 0.36036036, 0.36236236, 0.36236236, 0.36836837, 0.37137137, 0.37637638, 0.37637638, 0.37737738, 0.37937938, 0.37937938, 0.38038038, 0.38438438, 0.38738739, 0.39039039, 0.39039039, 0.39339339, 0.39339339, 0.39639640, 0.39839840, 0.40040040, 0.40540541, 0.40640641, 0.41141141, 0.41541542, 0.41641642, 0.41741742, 0.42042042, 0.42242242, 0.42742743, 0.43043043, 0.43343343, 0.43543544, 0.43943944, 0.44044044, 0.44244244, 0.44344344, 0.44444444, 0.44544545, 0.45045045, 0.45145145, 0.45245245, 0.45445445, 0.45545546, 0.45845846, 0.46046046, 0.46046046, 0.46046046, 0.46246246, 0.46746747, 0.46746747, 0.47047047, 0.47247247, 0.47347347, 0.47747748, 0.47947948, 0.48148148, 0.48248248, 0.48548549, 0.48948949, 0.49049049]

min_interior_cdf = [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00100100, 0.00600601, 0.00600601, 0.00700701, 0.00900901, 0.01201201, 0.01501502, 0.01601602, 0.01901902, 0.02602603, 0.03103103, 0.03403403, 0.03903904, 0.05205205, 0.06006006, 0.06906907, 0.07407407, 0.08508509, 0.09209209, 0.10410410, 0.11711712, 0.12512513, 0.14014014, 0.15815816, 0.17217217, 0.18818819, 0.20120120, 0.21021021, 0.21921922, 0.23323323, 0.25025025, 0.25825826, 0.26626627, 0.27627628, 0.28928929, 0.29629630, 0.31131131, 0.32232232, 0.33533534, 0.35235235, 0.35835836, 0.36836837, 0.37937938, 0.38938939, 0.39639640, 0.40940941, 0.41341341, 0.42442442, 0.42942943, 0.43543544, 0.44344344, 0.45045045, 0.45645646, 0.46746747, 0.47547548, 0.47847848, 0.48048048, 0.48648649, 0.49349349, 0.49849850, 0.50550551, 0.51251251, 0.51751752, 0.52352352, 0.52952953, 0.53353353, 0.53953954, 0.54454454, 0.54854855, 0.55455455, 0.55555556, 0.55955956, 0.56256256, 0.56656657, 0.57357357, 0.57957958, 0.58558559, 0.58858859, 0.58958959, 0.59459459, 0.59659660, 0.60060060, 0.60460460, 0.60860861, 0.60960961, 0.61661662, 0.62162162, 0.62462462, 0.62862863, 0.63463463, 0.63963964, 0.64264264, 0.64864865, 0.65265265, 0.65565566, 0.65665666, 0.65865866, 0.66566567, 0.66866867, 0.67167167, 0.67467467, 0.67667668, 0.67667668, 0.67767768, 0.68168168, 0.68868869, 0.69069069, 0.69469469, 0.69669670, 0.70170170, 0.70270270, 0.70470470, 0.70670671, 0.70870871, 0.70970971, 0.70970971, 0.71071071, 0.71571572, 0.71571572, 0.71571572, 0.72072072, 0.72372372, 0.72572573, 0.72972973, 0.73073073, 0.73373373, 0.73373373, 0.73773774, 0.74074074, 0.74174174, 0.74574575, 0.74774775, 0.74974975, 0.74974975, 0.75075075, 0.75075075, 0.75375375, 0.75475475, 0.75875876, 0.75975976, 0.75975976, 0.76276276, 0.76576577, 0.76676677, 0.77177177, 0.77277277, 0.77477477, 0.77577578, 0.77577578, 0.77677678, 0.77677678, 0.77677678, 0.78178178, 0.78178178, 0.78278278, 0.78278278, 0.78478478, 0.78778779, 0.79179179, 0.79279279, 0.79479479, 0.79479479, 0.79679680, 0.79679680, 0.79879880, 0.79879880, 0.79979980, 0.79979980, 0.80180180, 0.80380380, 0.80680681, 0.80980981, 0.80980981, 0.81081081, 0.81181181, 0.81181181, 0.81281281, 0.81281281, 0.81281281, 0.81481481, 0.81581582, 0.81581582, 0.81681682, 0.81681682, 0.81681682, 0.81781782, 0.81781782, 0.81881882, 0.82082082, 0.82182182]

min_sqp = [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00400400, 0.00600601, 0.02302302, 0.05005005, 0.08208208, 0.12612613, 0.17417417, 0.22522523, 0.28728729, 0.34834835, 0.39739740, 0.44044044, 0.48348348, 0.51751752, 0.55155155, 0.59959960, 0.63363363, 0.65965966, 0.68068068, 0.70370370, 0.71971972, 0.73073073, 0.75175175, 0.76676677, 0.77777778, 0.78878879, 0.80280280, 0.80980981, 0.81481481, 0.82182182, 0.82582583, 0.83083083, 0.83983984, 0.84884885, 0.85385385, 0.85585586, 0.86386386, 0.86586587, 0.87087087, 0.87487487, 0.88088088, 0.88388388, 0.88588589, 0.88788789, 0.89589590, 0.89589590, 0.90190190, 0.90490490, 0.90590591, 0.90690691, 0.90790791, 0.91591592, 0.91791792, 0.91991992, 0.92192192, 0.92392392, 0.92492492, 0.92792793, 0.92992993, 0.92992993, 0.93293293, 0.93293293, 0.93593594, 0.93693694, 0.93793794, 0.94094094, 0.94094094, 0.94194194, 0.94194194, 0.94294294, 0.94494494, 0.94494494, 0.94694695, 0.94694695, 0.94694695, 0.94694695, 0.94694695, 0.94894895, 0.94894895, 0.95095095, 0.95095095, 0.95095095, 0.95195195, 0.95295295, 0.95295295, 0.95295295, 0.95295295, 0.95395395, 0.95595596, 0.95595596, 0.95595596, 0.95695696, 0.95695696, 0.95695696, 0.95695696, 0.95695696, 0.95695696, 0.95695696, 0.95795796, 0.95795796, 0.95795796, 0.95795796, 0.95795796, 0.95795796, 0.95795796, 0.95795796, 0.95795796, 0.95895896, 0.95895896, 0.95895896, 0.95895896, 0.95895896, 0.95895896, 0.95895896, 0.95995996, 0.95995996, 0.95995996, 0.96096096, 0.96196196, 0.96196196, 0.96196196, 0.96296296, 0.96296296, 0.96496496, 0.96496496, 0.96496496, 0.96496496, 0.96596597, 0.96696697, 0.96696697, 0.96696697, 0.96696697, 0.96796797, 0.96796797, 0.96896897, 0.96896897, 0.96896897, 0.96996997, 0.97097097, 0.97097097, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97297297, 0.97397397, 0.97397397, 0.97397397, 0.97397397, 0.97397397, 0.97397397, 0.97397397, 0.97397397, 0.97397397, 0.97397397, 0.97497497, 0.97497497, 0.97497497, 0.97597598, 0.97597598, 0.97597598, 0.97597598, 0.97597598, 0.97597598, 0.97597598, 0.97597598, 0.97597598, 0.97597598, 0.97597598, 0.97597598, 0.97597598, 0.97597598, 0.97597598, 0.97697698]

#Generating CDF results of neural network, uniform, and greedy methods.
stats22 = np.arange(0, 0.2, 0.001)
ep_cdf22 =np.zeros([1,len(stats22)])[0]
unif_cdf22 =np.zeros([1,len(stats22)])[0]
const_cdf22 =np.zeros([1,len(stats22)])[0]
count22 = 0

for il in stats22:
  tata22 = len(np.argwhere(ep_reward_list<=il))
  tata222 = len(np.argwhere(unif_reward_list<=il))
  tata223 = len(np.argwhere(const_reward_list<=il))
  ep_cdf22[count22] = tata22/len(ep_reward_list)
  unif_cdf22[count22] = tata222/len(unif_reward_list)
  const_cdf22[count22] = tata223/len(const_reward_list)
  count22 = count22 + 1

#Plotting code
plt.plot(stats22*1000,ep_cdf22,linewidth=3)
plt.plot(stats22*1000,unif_cdf22,'--')
plt.plot(stats22*1000,const_cdf22,'--')
plt.plot(stats22*1000,min_interior_cdf,':')
plt.plot(stats22*1000,min_sqp,':')
plt.title('Comparison under Noisy CSI ($\sigma$ = 0.01) & $L_{1}^{Max}$ - 25 APs & 6 UEs')
plt.xlabel("Max Latency (ms)")
plt.ylabel("CDF")
plt.legend(['MPA-NN [This Paper]', 'Uniform Allocation', 'Greedy Allocation','Centralized Method [IP]','Centralized Method [SQP]'])
plt.grid()
plt.show()


print('ep reward list')
print(ep_reward_list)
print('mean of ep_reward_list')
print(np.mean(ep_reward_list))
print('std of ep_reward_list')
print(np.std(ep_reward_list))
print('unif reward list')
print(unif_reward_list)
print('mean of unif_reward_list')
print(np.mean(unif_reward_list))
print('std of unif_reward_list')
print(np.std(unif_reward_list))
print('const reward list')
print(const_reward_list)
print('mean of const_reward_list')
print(np.mean(const_reward_list))
print('std of const_reward_list')
print(np.std(const_reward_list))
print('*********')
print('*********')
print('ep cdf22')
print(ep_cdf22)
print('unif cdf22')
print(unif_cdf22)
print('const cdf22')
print(const_cdf22)
