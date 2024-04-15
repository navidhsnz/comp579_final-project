import  gymnasium as gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from memory import PrioritizedReplayMemory, RandomReplayMemory

from train import Agent

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

if __name__ == "__main__":
    model_parameters = {
        "gamma": 0.99,
        "lr": 5e-4,
        "TAU": 0.01,
        "episode_num": 400,
        "batch_size": 64,
        "epsilon_max": 0.1,
        "epsilon_min": 0.02,
        "max_episode_len": -1,
        "device": device,
        "call_env_changes": False,
        "env_changes": lambda *_: None,
        "env": gym.make("CartPole-v1", render_mode='human'),
        "buffer": RandomReplayMemory(buffer_size= 50_000, device = device),
        # "buffer": PrioritizedReplayMemory(buffer_size= 50_000, device = device, alpha=0.7, beta=0.2)
        }
    
    agent = Agent(**model_parameters)
    agent.train()

    