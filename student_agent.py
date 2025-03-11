# Remember to adjust your student ID in meta.xml

# obs = (
#     taxi_row, taxi_col,  
#     station_0_x, station_0_y,  
#     station_1_x, station_1_y,  
#     station_2_x, station_2_y,  
#     station_3_x, station_3_y,  
#     obstacle_north, obstacle_south, obstacle_east, obstacle_west,  
#     passenger_look, destination_look
# )

import numpy as np
import pickle
import random
import gym

# with open("data.pkl", "rb") as f:
#     Q_table = pickle.load(f)

# stage = 0
# substage = 0

# def comp_diff(x, y):
#     ret = 0
#     if x > y: ret = 1
#     elif x < y: ret = -1
#     return ret

# def refine_obs(obs, stage, substage):
#     taxi_x = obs[0]
#     taxi_y = obs[1]
    
#     new_obs = [0] * 5
    
#     if(substage == 0):
#         new_obs[0] = (comp_diff(taxi_x, obs[2]), comp_diff(taxi_y, obs[3]))
#     elif(substage == 1):
#         new_obs[0] = (comp_diff(taxi_x, obs[4]), comp_diff(taxi_y, obs[5]))
#     elif(substage == 2):
#         new_obs[0] = (comp_diff(taxi_x, obs[6]), comp_diff(taxi_y, obs[7]))
#     else:
#         new_obs[0] = (comp_diff(taxi_x, obs[8]), comp_diff(taxi_y, obs[9]))
        
#     new_obs[1:5] = obs[10:14]
    
#     return tuple(new_obs)

# def get_action(obs):
    
#     # TODO: Train your own agent
#     # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
#     # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
#     #       To prevent crashes, implement a fallback strategy for missing keys. 
#     #       Otherwise, even if your agent performs well in training, it may fail during testing.
#     global stage
#     global substage
#     if stage is None:
#         stage = 0
#     if substage is None:
#         substage = 0
    
#     reached = 0
#     if(stage < 2):
#         reached = obs[14]
#     else:
#         reached = obs[15]
    
#     temp_obs = refine_obs(obs, stage, substage)
    
#     if(stage == 0):
#         if(temp_obs[0] == (0, 0)):
#             if(reached == 1):
#                 stage = 1
#             else:
#                 substage += 1
#     elif(stage == 1):
#         stage = 2
#         substage = 0
#     elif(stage == 2):
#         if(temp_obs[0] == (0, 0)):
#             if(reached == 1):
#                 stage = 3
#             else:
#                 substage += 1
    
#     ref_obs = refine_obs(obs, stage, substage)

#     action = 0
#     if stage == 1:
#         action = 4
#     elif stage == 3:
#         action = 5
#     else:
#         if(ref_obs in Q_table):
#             action = np.argmax(Q_table[ref_obs])
#         else:
#             action = random.choice([0, 1, 2, 3])
    
#     return action

last = 0

def get_action(obs):
    global last
    if last is None:
        last = 0
    
    if last == 0:
        last = 1
    else:
        last = 0
    
    return last

