import gymnasium as gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from DQN import DQN
from mem.buffer import ReplayBuffer, PrioritizedReplayBuffer

class Agent:
    def __init__(self, **kwargs):
        self.gamma = kwargs.get("gamma")
        self.lr = kwargs.get("lr")
        self.tau = kwargs.get("tau")
        self.TAU = self.tau
        self.ep_num = kwargs.get("ep_num")
        self.batch_size = kwargs.get("batch_size")
        self.eps_max = kwargs.get("eps_max")
        self.eps_min = kwargs.get("eps_min")
        self.buffer = kwargs.get("buffer")
        self.env = kwargs.get("env")
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr) #W and amsgrad=True
    

    def select_action(self, state, episode):
        eps = self.eps_max-((episode/self.ep_num)*(self.eps_max-self.eps_min))
        if random.random() < eps:
            action = self.env.action_space.sample()
        else:
            state = torch.as_tensor(state, dtype=torch.float).to(self.device)
            action = torch.argmax(self.policy_net(state)).cpu().numpy().item()
        return action

    def train(self):
        step=0
        list_ep_lens = []
        for episode in range(self.ep_num):
            state, _ = self.env.reset()
            done = False
            ep_len=0
            while not done:
                ep_len+=1
                step+=1
                action = self.select_action(state, episode)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.buffer.add((state, action, reward, next_state, int(done)))
                state = next_state

                if step > self.batch_size:
                    if isinstance(self.buffer, ReplayBuffer):
                        batch = self.buffer.sample(self.batch_size)
                        _, td_error = self.update(batch)
                    elif isinstance(self.buffer, PrioritizedReplayBuffer):
                        batch, weights, tree_idxs = self.buffer.sample(self.batch_size)
                        _, td_error = self.update(batch, weights=weights)
                        self.buffer.update_priorities(tree_idxs, td_error.cpu().numpy())

            print(episode, ep_len)
            list_ep_lens.append(ep_len)

    def update(self, batch, weights=None):
        state, action, reward, next_state, done = batch
        Q_next = self.target_net(next_state.to(self.device)).max(1).values
        Q_target = reward.to(self.device) + self.gamma * (1 - done.to(self.device)) * Q_next
        Q = self.policy_net(state.to(self.device))[torch.arange(len(action)), action.to(torch.long).flatten()]

        assert Q.shape == Q_target.shape, f"{Q.shape}, {Q_target.shape}"

        if weights is None:
            weights = torch.ones_like(Q)

        td_error = torch.abs(Q - Q_target).detach()
        loss = torch.mean((Q - Q_target)**2 * weights.to(self.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                    1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)


        return loss.item(), td_error
