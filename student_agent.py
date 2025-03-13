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

# with open("data.pkl", "rb") as f:
#     Q_table = pickle.load(f)

# stage = 0
# substage = 0
# past_obs = np.zeros(9)
# destiny = -1

def random_pick(obs):
    possible_actions = [0, 1, 2, 3]
                
    if obs[10]:
        possible_actions.remove(1)
    if obs[11]:
        possible_actions.remove(0)
    if obs[12]:
        possible_actions.remove(2)
    if obs[13]:
        possible_actions.remove(3)
    
    if possible_actions:
        action = random.choice(possible_actions)
    
    return action

# def comp_diff(x, y):
#     ret = 0
#     # if x > y: ret = 1
#     # elif x < y: ret = -1
#     ret = x - y
#     return ret

# def refine_obs(obs, stage, substage, past_obs):
#     taxi_x = obs[0]
#     taxi_y = obs[1]
    
#     new_obs = [0] * 9
    
#     if(substage == 0):
#         new_obs[0] = (comp_diff(taxi_x, obs[2]), comp_diff(taxi_y, obs[3]))
#     elif(substage == 1):
#         new_obs[0] = (comp_diff(taxi_x, obs[4]), comp_diff(taxi_y, obs[5]))
#     elif(substage == 2):
#         new_obs[0] = (comp_diff(taxi_x, obs[6]), comp_diff(taxi_y, obs[7]))
#     else:
#         new_obs[0] = (comp_diff(taxi_x, obs[8]), comp_diff(taxi_y, obs[9]))
        
#     new_obs[1:5] = obs[10:14]
    
#     new_obs[5:9] = past_obs[1:5]
    
#     return tuple(new_obs)

# def get_action(obs):
    
#     # TODO: Train your own agent
#     # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
#     # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
#     #       To prevent crashes, implement a fallback strategy for missing keys. 
#     #       Otherwise, even if your agent performs well in training, it may fail during testing.
#     global stage
#     global substage
#     global past_obs
#     global destiny
#     if stage is None:
#         stage = 0
#     if substage is None:
#         substage = 0
#     if past_obs is None:
#         past_obs = np.zeros(9)
#     if destiny is None:
#         destiny = -1
    
#     reached1 = obs[14]
#     reached2 = obs[15]
    
#     temp_obs = refine_obs(obs, stage, substage, past_obs)
    
#     possible_actions = [0, 1, 2, 3]
                
#     if obs[10]:
#         possible_actions.remove(1)
#     if obs[11]:
#         possible_actions.remove(0)
#     if obs[12]:
#         possible_actions.remove(2)
#     if obs[13]:
#         possible_actions.remove(3)
    
#     if(stage == 0):
#         if(temp_obs[0] == (0, 0)):
#             if(reached2 == 1):
#                 destiny = substage
#             if(reached1 == 1):
#                 stage = 1
#             else:
#                 substage += 1
#     elif(stage == 1):
#         stage = 2
#         if(destiny != -1):
#             substage = destiny
#     elif(stage == 2):
#         if(temp_obs[0] == (0, 0)):
#             if(reached2 == 1):
#                 stage = 3
#             else:
#                 substage += 1
    
#     ref_obs = refine_obs(obs, stage, substage, past_obs)

#     action = 0
#     if stage == 1:
#         action = 4
#     elif stage == 3:
#         action = 5
#     else:
#         if(ref_obs in Q_table):
#             action = np.argmax(Q_table[ref_obs])
#             if (np.random.rand() < 0.1) or (action not in possible_actions) or action == 5 or action == 4:
#                 action = random_pick(obs)
#         else:
#             action = random_pick(obs)
    
#     past_obs = obs
    
#     return action

def get_action(obs):
    return random_pick(obs)
import gym
import torch

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    
    action = [0,1, 2, 3, 4, 5]
    if random.uniform(0, 1) < 0.25:
        return random.choice(action)
    else:
        if obs[10]==1:
            action.remove(1)  
        if obs[13]==1:
            action.remove(3)  
        if obs[12]==1:
            action.remove(2)
        if obs[11]==1:
            action.remove(0)
        if obs[14]!=1:
            action.remove(4)
        if obs[15]!=1:
            action.remove(5)
        return random.choice(action)
            
    #return random.choice(action) # Choose a random action

    #return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

