import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
from simple_custom_taxi_env import SimpleTaxiEnv
import random
import pickle

def comp_diff(x, y):
    ret = 0
    # if x > y: ret = 1
    # elif x < y: ret = -1
    ret = x - y
    return ret

# stage 0: find passenger
# 0-0: check R
# 0-1: check G
# 0-2: check Y
# 0-3: check B
# if located: 
# stage 1: pickup passenger

# stage 2: find destination
# 2-0: check R
# 2-1: check G
# 2-2: check Y
# 2-3: check B
# if located: 
# stage 3: dropoff passenger

# state:
# target relative position
# last action
# walls * 4

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
    
    if obs[2]==1 or last_action==0:
        possible_actions.remove(1)  
    if obs[3]==1 or last_action==1:
        possible_actions.remove(0)
    if obs[4]==1 or last_action==3:
        possible_actions.remove(2)
    if obs[5]==1 or last_action==2:
        possible_actions.remove(3)  
    
    if possible_actions:
        action = random.choice(possible_actions)
    
    return action

def refine_obs(obs, stage, substage, past_obs, last_action, pickup):
    taxi_x = obs[0]
    taxi_y = obs[1]
    
    new_obs = [0] * 7
    
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
    
    # new_obs[6:11] = past_obs[1:6]
    
    return tuple(new_obs)

def train_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(**env_config)
    
    q_table = {}
    episodes = 500000
    epsilon = 1
    rewards_per_episode = []
    
    alpha = 0.01
    gamma = 0.99
    epsilon_end = 0.1
    decay_rate = 0.999995
    doneCnt = 0
    pickCnt = 0
    hit_wall = 0
    bad_drop = 0
    
    for episode in range(episodes):
        obs, _ = env.reset()
        env.place_random_obstacles(15)

        total_reward = 0
        done = False
        step_count = 0
        prev_pickup = False
        
        stage = 0
        substage = 0
        destiny = -1
        last_action = 0
        
        obs = refine_obs(obs, stage, substage, np.zeros(9), 0, prev_pickup)
        q_table[obs] = np.array([0.0, 0.0, 0.0, 0.0])
    
        while not done:
            
            if np.random.rand() < epsilon:
                action = random_pick(obs, last_action)
            else:
                action = np.argmax(q_table[obs])
                
            if stage == 1:
                action = 4
            elif stage == 3:
                action = 5
            
            next_obs, reward, done, _ = env.step(action)
            last_action = action
            
            if(action < 4 and reward == -5.1):
                reward = -100.0
                hit_wall += 1
            
            reached1 = next_obs[14]
            reached2 = next_obs[15]
            
            temp_obs = refine_obs(next_obs, stage, substage, obs, last_action, prev_pickup)
            step_count += 1

            if(stage == 0):
                if(temp_obs[0] == (0, 0)):
                    reward += 100.0
                    if(reached2 == 1):
                        destiny = substage
                    if(reached1 == 1):
                        stage = 1
                    else:
                        substage += 1
                if(action == 4 or action == 5):
                    reward -= 10000.0
                    bad_drop += 1
            elif(stage == 1):
                if not env.passenger_picked_up:
                    break
                stage = 2
                if(destiny != -1):
                    substage = destiny
            elif(stage == 2):
                if(temp_obs[0] == (0, 0)):
                    reward += 100.0
                    if(reached2 == 1):
                        stage = 3
                    else:
                        substage += 1
                if(action == 4 or action == 5):
                    reward -= 10000.0
                    bad_drop += 1
            
            if(action == 5 and not done):
                reward -= 10000.0
                bad_drop += 1
            
            reward -= 0.01
            
            if (not prev_pickup) and env.passenger_picked_up:
                prev_pickup = True
                reward += 200.0
                pickCnt += 1
            
            if(done):
                reward += 300.0
                doneCnt += 1
                
            next_obs = refine_obs(next_obs, stage, substage, obs, last_action, prev_pickup)
            
            total_reward += reward
            
            if next_obs not in q_table:
                if(stage == 1 or stage == 3):
                    q_table[next_obs] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    q_table[next_obs] = np.array([0.0, 0.0, 0.0, 0.0])

            # TODO: Apply the Q-learning update rule (Bellman equation).
            q_table[obs][action] = (1 - alpha) * q_table[obs][action] + \
                                    alpha * (reward + gamma * np.max(q_table[next_obs]))

            # TODO: Update the state to the next state.
            obs = next_obs
            
            if (step_count == 1000):
                break
            
        rewards_per_episode.append(total_reward)

        # TODO: Decay epsilon over time to gradually reduce exploration.
        epsilon = max(epsilon_end, epsilon * decay_rate)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}")
            print("score:", pickCnt, doneCnt, hit_wall, bad_drop)
            pickCnt = 0
            doneCnt = 0
            hit_wall = 0
            bad_drop = 0
    
    with open("data.pkl", "wb") as file:
        pickle.dump(q_table, file)
        
    return total_reward

if __name__ == "__main__":
    env_config = {
        "fuel_limit": 5000
    }
    
    train_agent("student_agent.py", env_config, render=False)
    