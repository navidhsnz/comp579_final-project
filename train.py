# This code is written for our final project of comp 579 in Winter 2024.
# The code in this file handles the training of the models. We implement a class called agent and receives all the parameters in its initialization. This class contains important functions such as:

# trian: takes care of training the agent. It mainly involves performaing episodes by taking a step in environment and adding the transition to the assigned memory buffer.

# select_action: decides an action to take based on epsilon greedy policy.

# get_reduced_epsilon: calculates a reduced epsilon value based on a give episode number.

# sample_and_update: this funciton will call the meomory sample function to get a batch of transitions and call the 'update' function within the same class to update the neural network based on the obtained batch.
# -----------------------------------------------------------------------
import gymnasium as gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from DQN import DQN
from memory import PrioritizedReplayMemory, RandomReplayMemory
from tqdm import tqdm


class Agent:
    # Initializes the reinforcement learning agent.
    def __init__(self, **kwargs):
        # Initialize hyperparameters and environment-related variables
        self.gamma = kwargs.get("gamma")
        self.lr = kwargs.get("lr")
        self.tau = kwargs.get("TAU")
        self.TAU = self.tau
        self.ep_num = kwargs.get("episode_num")
        self.batch_size = kwargs.get("batch_size")
        self.eps_max = kwargs.get("epsilon_max")
        self.eps_min = kwargs.get("epsilon_min")
        self.buffer = kwargs.get("buffer")
        self.max_episode_len = kwargs.get("max_episode_len")
        self.env = kwargs.get("env")
        self.non_stationarity = kwargs.get("non_stationarity")
        self.env_changes = kwargs.get("env_changes")
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.device = kwargs.get("device")

        # Initialize the policy and target networks
        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)  # W and amsgrad=True

    # Calculates the reduced epsilon value based on the current episode number.
    def get_reduced_epsilon(self, episode):
        return self.eps_max - ((episode / self.ep_num) * (self.eps_max - self.eps_min))

    # Selects an action based on epsilon-greedy policy.
    def select_action(self, state, episode):
        eps = self.get_reduced_epsilon(episode)
        if random.random() < eps:
            action = self.env.action_space.sample()
        else:
            state = torch.as_tensor(state, dtype=torch.float).to(self.device)
            action = torch.argmax(self.policy_net(state)).cpu().numpy().item()
        return action

    # Trains the agent using the specified environment and hyperparameters. 
    # this function involves perfoming each episode and adding transition information to buffer and also calling helper function to sample and train the neural network.
    def train(self):
        step = 0
        list_ep_lens = []
        list_loss_vals = []
        list_accumulated_reward = []
        # this loop will perform the episodes
        for episode in range(self.ep_num):
            state = self.env.reset()[0]
            ep_len = 0
            done = False
            if self.non_stationarity:
                    self.env_changes(self.env, episode) # this fuction changes the environment based on a given episode number. The fuction is defined and given as one of the parameters of the agent. It is used for the case of non-stationary environments as well as for rendering a visualization of the environment when needed.
            accumulated_reward = 0
            reduced_gamma = 1
            while not done: # this loops follows the steps within an episode untill termination
                # Here is to end the episode forcefully if it goes above teh max_episode_len
                if ep_len==self.max_episode_len:
                    break 

                ep_len += 1
                action = self.select_action(state, episode)
                next_state, reward, terminated, truncated, _ = self.env.step(action) # take a step
                done = terminated or truncated
                self.buffer.add((state, action, reward, next_state, int(done))) # add transition to buffer
                state = next_state
                accumulated_reward += (reward * reduced_gamma)
                reduced_gamma*= self.gamma
                loss_val = 0

                if step > self.batch_size: # if the step is larger than batch_size, we start sampling and training. It is used to avoid staring the trainig before there is adequate data in the buffer.
                    loss_val = self.sample_and_update()
                else:
                    step+=1 # this variable is used to keep track of how many steps went by without peroforming a training. It is used to avoid staring the trainig before there is adequate data in the buffer.

            print(f"episode:{episode}, ep_length:{ep_len}, acc_reward:{accumulated_reward}, loss:{loss_val}")
            list_ep_lens.append(ep_len)
            list_loss_vals.append(loss_val)
            list_accumulated_reward.append(accumulated_reward)

        return list_ep_lens, list_loss_vals, list_accumulated_reward

    # This funciton samples transitions from the replay buffer and calls a helper function to update the neural network weigts.
    def sample_and_update(self):
        batch, sample_weights, indices = self.buffer.sample(self.batch_size)
        loss_val , td_error = self.update(batch, sample_weights)
        self.buffer.update_priorities(indices, td_error.cpu().numpy()) # this function is defined to do nothing in random buffer implementaiton.
        return loss_val
    
    # This funciton updates the neural networks using gradient descent. We use a target and main neural network for better stability.
    def update(self, batch, weights):
        state, action, reward, next_state, done = batch
        Q_next = self.target_net(next_state.to(self.device)).max(1).values
        Q_target = reward.to(self.device) + self.gamma * (1 - done.to(self.device)) * Q_next
        Q = self.policy_net(state.to(self.device))[torch.arange(len(action)), action.to(torch.long).flatten()]

        assert Q.shape == Q_target.shape, f"wrong shape: {Q.shape} not the same as {Q_target.shape}"
        
        # we calculate TD error and loss
        td_error = torch.abs(Q - Q_target).detach()
        loss = torch.mean((Q - Q_target) ** 2 * weights.to(self.device))

        # We perform optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

         # We update target network parameters
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

        return loss.item(), td_error
