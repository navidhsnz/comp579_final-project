# This code is part of our final project of comp 579 in Winter 2024.
# We used the standard pytorch recommendation for initializing the DQN class below.
# This part of the code is based on the tutorial in following link: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    # Initializes a Deep Q-Network (DQN) model
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    # This fuction defines the forward pass of the DQN model by combining the layers and adding non-linearity (ReLu)
    def forward(self, x):
        x = F.relu(self.layer1(x)) # # We apply the ReLU activation to the first two layers' outputs
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

