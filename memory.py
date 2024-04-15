# This code is written for our final project of comp 579 in Winter 2024.
# This file contains the implementation of two memeory buffer algorithms in the form of classes. 
# First one is Prioritized Replay Memeory from the paper: https://arxiv.org/abs/1511.05952v4.
# Second is the random replay buffer which uniformly selects a batch experiences to train the neural network.

import torch
import random
import numpy as np
from collections import deque

# this class implements prioritized experinece replay
class PrioritizedReplayMemory:
    #Initializes the prioritized experinece replay buffer.
    def __init__(self, buffer_size, device, alpha=0.1, beta=0.1, eps=1e-2):
        self.data = deque(maxlen=buffer_size) # we use a deque data structure for storing the data. 
        self.eps = eps
        self.alpha = alpha 
        self.beta = beta 
        self.max_priority = eps
        self.device = device
        self.count = 0
        self.current_size = 0
        self.size = buffer_size

    # Adds a new transition (experience) to the buffer. The max buffer size was passed to the deque data structure in its initializaion. It automatically handles the storage when going above limit by removing the oldest data to make space for newer ones.
    def add(self, transition):
        transition_on_tensor = [torch.as_tensor(t,device=self.device) for t in transition]
        self.data.append([transition_on_tensor,self.max_priority])
        self.current_size = min(self.size, self.current_size + 1) # the variable current_size is to know how many transitions are stored in the memory. if we go above max limit (buffer size), the data strucuture deque automatically makes space for newer data by removing the oldes ones, which means the current_size won't increase any further than self.size

    # samples a batch of transitions from the replay memory with their probabilities. Also, importance sampling weights are computed. These parameters are returned to be used for the update step.
    def sample(self, batch_size):
        assert self.current_size >= batch_size, "batch size is larger than buffer size"
        sampled_w_indx = random.sample(list(enumerate(self.data)),batch_size)
        indexes = np.array([single_sample[0] for single_sample in sampled_w_indx ])
        priorities = np.array([single_sample[1][1] for single_sample in sampled_w_indx ])
        transitions = [single_sample[1][0] for single_sample in sampled_w_indx ]
        probs = priorities / sum(priorities)
        weights = (self.current_size*probs)**-self.beta
        weights = weights / weights.max()
        weights = torch.from_numpy(np.array(weights))
        batch = tuple( torch.from_numpy(np.array([transition[i].cpu() for transition in transitions])).to(self.device) for i in range(5))
        return batch, weights, indexes
    
    #Updates the priorities of transitions in the replay memory based on their temporal difference (TD) errors. Each transition's priority is calculated using its TD error and a small epsilon value (pre-defined). The maximum priority in the memory is also updated if a new higher priority is encountered.
    def update_priorities(self, data_idxs, td_diffs):
        for data_idx, td_diff in zip(data_idxs, td_diffs):
            priority = (abs(td_diff) + self.eps) ** self.alpha
            self.data[data_idx][1] = priority 
            self.max_priority = max(self.max_priority, priority)

# This class implemtns a random replay memory buffer where transitions are sampled uniformly.
class RandomReplayMemory:
    #Initializes the random replay memory buffer.
    def __init__(self, buffer_size, device):
        self.data = deque(maxlen=buffer_size) # we use a deque data structure for storing the data. 
        self.device = device
        self.current_size = 0
        self.size = buffer_size

    # Adds a new transition to the replay memory. The max buffer size was passed to the deque data structure in its initializaion. It automatically handles the storage when going above limit by removing the oldest data to make space for newer ones.
    def add(self, transition):
        self.data.append(transition)
        self.current_size = min(self.size, self.current_size + 1)

    # This fuction samples a batch of transitions from the replay memory.
    def sample(self, batch_size):
        assert self.current_size >= batch_size, "batch size is larger than buffer size"
        sample_idxs = np.random.choice(self.current_size, batch_size, replace=False)
        sample_data_list = [self.data[i] for i in sample_idxs]
        batch = tuple(torch.from_numpy(np.array([sample_data[i] for sample_data in sample_data_list])).to(self.device) for i in range(5))
        weights = torch.ones(len(sample_idxs))
        return batch, weights, sample_idxs
    
    # this is a placeholder function for updating priorities, and it is not used for random replay. However we added it to simplify the usage of the memeory buffer classes in the rest of the project.
    def update_priorities(self, data_idxs, td_diffs):
        pass

