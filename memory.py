import torch
import random
import numpy as np
from collections import deque

class PrioritizedReplayMemory:
    def __init__(self, buffer_size, device, alpha=0.1, beta=0.1, eps=1e-2):
        self.data = deque(maxlen=buffer_size)
        self.eps = eps
        self.alpha = alpha 
        self.beta = beta 
        self.max_priority = eps
        self.device = device
        self.count = 0
        self.current_size = 0
        self.size = buffer_size

    def add(self, transition):
        transition_on_tensor = [torch.as_tensor(t,device=self.device) for t in transition]
        self.data.append([transition_on_tensor,self.max_priority])
        self.current_size = min(self.size, self.current_size + 1)

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
    
    def update_priorities(self, data_idxs, td_diffs):
        for data_idx, td_diff in zip(data_idxs, td_diffs):
            priority = (abs(td_diff) + self.eps) ** self.alpha
            self.data[data_idx][1] = priority 
            self.max_priority = max(self.max_priority, priority)


class RandomReplayMemory:
    def __init__(self, buffer_size, device):
        self.data = deque(maxlen=buffer_size)
        self.device = device
        self.current_size = 0
        self.size = buffer_size

    def add(self, transition):
        self.data.append(transition)
        self.current_size = min(self.size, self.current_size + 1)

    def sample(self, batch_size):
        assert self.current_size >= batch_size, "batch size is larger than buffer size"
        sample_idxs = np.random.choice(self.current_size, batch_size, replace=False)
        sample_data_list = [self.data[i] for i in sample_idxs]
        batch = tuple(torch.from_numpy(np.array([sample_data[i] for sample_data in sample_data_list])).to(self.device) for i in range(5))
        weights = torch.ones(len(sample_idxs))
        return batch, weights, sample_idxs
    
    def update_priorities(self, data_idxs, td_diffs):
        pass

