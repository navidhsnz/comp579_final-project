import  gymnasium as gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from tqdm import tqdm
from memory import RandomReplayMemory, PrioritizedReplayMemory
from train import Agent
from pathos.multiprocessing import ProcessingPool as Pool
from gridWorldEnv import GridWorldEnv
import matplotlib.pyplot as plt
import time

device = torch.device("cpu")



def env_changes(env, ep_num):
    if ep_num in [298, 698, 998]: #[20, 100, 250, 300, 301, 302, 310, 400, 500, 690, 700, 701, 702, 703, 710, 900, 998]:
        env.render_mode= "human"
    else:
        env.render_mode=None

    if ep_num == 0:
        env.switch_doors(top_door="open",bottom_door="close")
        print("door_changed")
    elif ep_num == 300:
        env.switch_doors(top_door="close",bottom_door="open")
        print("door_changed")
    elif ep_num == 700:
        env.switch_doors(top_door="open",bottom_door="close")
        print("door_changed")

parameters1 = {
        "gamma": 0.99,
        "lr": 5e-4,
        "TAU": 0.01,
        "episode_num": 1000,
        "batch_size": 64,
        "epsilon_max": 0.1,
        "epsilon_min": 0.1,
        "device": device,
        "max_episode_len": 200,
        "non_stationarity": True,
        "env_changes": env_changes,
        "env": GridWorldEnv(size=10), #render_mode= "human", 
        # "env":  gym.make("CartPole-v1"),
        # "buffer": RandomReplayMemory(buffer_size= 5_000, device = device),
        "buffer": PrioritizedReplayMemory(buffer_size= 200_000, device = device, alpha=0.7, beta=0.2)
        }

# env = parameters1['env']
# env2 = gym.make("CartPole-v1")
# st , _ = env.reset()

# print(st)
# time.sleep(3)
# env.switch_doors(1)
# env.step(0)
# time.sleep(3)
# env.switch_doors(0)
# env.step(0)
# time.sleep(3)
# env.switch_doors(1)
# env.step(0)

agent = Agent(**parameters1)
ep_len_1, loss_vals_1 = agent.train()


parameters2 = {
        "gamma": 0.99,
        "lr": 5e-4,
        "TAU": 0.01,
        "episode_num": 1000,
        "batch_size": 64,
        "epsilon_max": 0.1,
        "epsilon_min": 0.1,
        "device": device,
        "max_episode_len": 200,
        "non_stationarity": True,
        "env_changes": env_changes,
        "env": GridWorldEnv(size=10), #render_mode= "human", 
        # "env":  gym.make("CartPole-v1"),
        "buffer": RandomReplayMemory(buffer_size= 200_000, device = device),
        # "buffer": PrioritizedReplayMemory(buffer_size= 200_000, device = device, alpha=0.7, beta=0.2)
        }

agent = Agent(**parameters2)
ep_len_2, loss_vals_2 = agent.train()


plt.plot(ep_len_1,label="random_memory")
plt.plot(ep_len_2,label="prioritized_memory")
plt.legend()
plt.show()
plt.plot(loss_vals_1,label="random_memory")
plt.plot(loss_vals_2,label="prioritized_memory")
plt.legend()
plt.show()