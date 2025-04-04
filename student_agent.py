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

with open("data.pkl", "rb") as f:
    Q_table = pickle.load(f)

stage = None
substage = None
past_obs = None
destiny = None
prev_pickup = None
last_action = None
visited = None

def random_pick(obs, last_action):
    possible_actions = [0, 1, 2, 3]
    
    action = 0
    if last_action==0:
        action = 1
    if last_action==1:
        action = 0
    if last_action==3:
        action = 2
    if last_action==2:
        action = 3
    
    if obs[10]==1 or last_action==0:
        possible_actions.remove(1)  
    if obs[11]==1 or last_action==1:
        possible_actions.remove(0)
    if obs[12]==1 or last_action==3:
        possible_actions.remove(2)
    if obs[13]==1 or last_action==2:
        possible_actions.remove(3)  
    
    if possible_actions:
        action = random.choice(possible_actions)
    
    return action

def comp_diff(x, y):
    ret = 0
    if x > y: ret = 1
    elif x < y: ret = -1
    return ret

def find_closest(obs, visited):
    taxi_row, taxi_col = obs[0], obs[1]
    min_dist = float('inf')
    closest_station = None

    for i in range(4):
        station_x, station_y = obs[2 + i * 2], obs[3 + i * 2]
        if visited[i] != 1:
            dist = abs(taxi_row - station_x) + abs(taxi_col - station_y)
            if dist < min_dist:
                min_dist = dist
                closest_station = i

    return closest_station

def refine_obs(obs, stage, substage, past_obs, last_action, pickup):
    taxi_x = obs[0]
    taxi_y = obs[1]
    
    new_obs = [0] * 11
    
    if(substage == 0):
        new_obs[0] = (comp_diff(taxi_x, obs[2]), comp_diff(taxi_y, obs[3]))
    elif(substage == 1):
        new_obs[0] = (comp_diff(taxi_x, obs[4]), comp_diff(taxi_y, obs[5]))
    elif(substage == 2):
        new_obs[0] = (comp_diff(taxi_x, obs[6]), comp_diff(taxi_y, obs[7]))
    else:
        new_obs[0] = (comp_diff(taxi_x, obs[8]), comp_diff(taxi_y, obs[9]))
    
    new_obs[1] = last_action
    
    new_obs[2:6] = obs[10:14]
    
    new_obs[6] = pickup
    
    new_obs[7:11] = past_obs[2:6]
    
    return tuple(new_obs)

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    global stage
    global substage
    global past_obs
    global destiny
    global last_action
    global prev_pickup
    global visited
    if stage is None:
        stage = 0
    if past_obs is None:
        past_obs = np.zeros(11)
    if destiny is None:
        destiny = -1
    if prev_pickup is None:
        prev_pickup = 0
    if last_action is None:
        last_action = 0
    if visited is None:
        visited = np.zeros(4)
    if substage is None:
        substage = find_closest(obs, visited)
    
    reached1 = obs[14]
    reached2 = obs[15]
    
    temp_obs = refine_obs(obs, stage, substage, past_obs, last_action, prev_pickup)
    
    if(stage == 0):
        if(temp_obs[0] == (0, 0)):
            visited[substage] = 1
            if(reached2 == 1):
                destiny = substage
            if(reached1 == 1):
                stage = 1
            else:
                substage = find_closest(obs, visited)
    elif(stage == 1):
        stage = 2
        if(destiny != -1):
            substage = destiny
        prev_pickup = 1
    elif(stage == 2):
        if(temp_obs[0] == (0, 0)):
            visited[substage] = 1
            if(reached2 == 1):
                stage = 3
            else:
                substage = find_closest(obs, visited)
    
    ref_obs = refine_obs(obs, stage, substage, past_obs, last_action, prev_pickup)

    action = 0
    if(ref_obs in Q_table):
        action = np.argmax(Q_table[ref_obs])
        if (np.random.rand() < 0.25 and action != 4 and action != 5): 
            action = random_pick(obs, last_action)
    else:
        action = random_pick(obs, last_action)
    
    past_obs = ref_obs
    last_action = action
    return action
