import gymnasium as gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from mem.buffer import ReplayBuffer, PrioritizedReplayBuffer

from train import Agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parameters = {
        "gamma": 0.99,
        "lr": 1e-4,
        "tau": 0.01,
        "ep_num": 500,
        "batch_size": 64,
        "eps_max": 0.7,
        "eps_min": 0.05,
        "env": gym.make("CartPole-v1"),
        # "buffer": ReplayBuffer(state_size=4, action_size=1, buffer_size= 50_000),
        "buffer": PrioritizedReplayBuffer(state_size=4, action_size=1,
                                          buffer_size= 50_000, alpha=0.7, beta=0.4)
        }
    
    agent = Agent(**parameters)
    steps = agent.train()

    