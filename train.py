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
    def __init__(self, **kwargs):
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

        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)  # W and amsgrad=True


    def get_reduced_epsilon(self, episode):
        return self.eps_max - ((episode / self.ep_num) * (self.eps_max - self.eps_min))


    def select_action(self, state, episode):
        eps = self.get_reduced_epsilon(episode)
        if random.random() < eps:
            action = self.env.action_space.sample()
        else:
            state = torch.as_tensor(state, dtype=torch.float).to(self.device)
            action = torch.argmax(self.policy_net(state)).cpu().numpy().item()
        return action

    def train(self):
        step = 0
        list_ep_lens = []
        list_loss_vals = []
        for episode in tqdm(range(self.ep_num)):
            state = self.env.reset()[0]
            ep_len = 0
            done = False
            if self.non_stationarity:
                    self.env_changes(self.env, episode)

            while not done:
                if ep_len>self.max_episode_len:
                    break
                ep_len += 1
                action = self.select_action(state, episode)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.buffer.add((state, action, reward, next_state, int(done)))
                state = next_state

                loss_val = 0

                if step > self.batch_size:
                    loss_val = self.sample_and_update()
                else:
                    step+=1


            # print(episode, ep_len)
            # print(loss_val)
            list_ep_lens.append(ep_len)
            list_loss_vals.append(loss_val)
            # if episode>50:
            #     self.env.render_mode = "human"
        return list_ep_lens, list_loss_vals

    def sample_and_update(self):
        batch, sample_weights, indices = self.buffer.sample(self.batch_size)
        loss_val , td_error = self.update(batch, sample_weights)
        self.buffer.update_priorities(indices, td_error.cpu().numpy()) # this function is defined to do nothing in random buffer implementaiton.
        return loss_val
    
    def update(self, batch, weights):
        state, action, reward, next_state, done = batch
        Q_next = self.target_net(next_state.to(self.device)).max(1).values
        Q_target = reward.to(self.device) + self.gamma * (1 - done.to(self.device)) * Q_next
        Q = self.policy_net(state.to(self.device))[torch.arange(len(action)), action.to(torch.long).flatten()]

        assert Q.shape == Q_target.shape, f"wrong shape: {Q.shape} not the same as {Q_target.shape}"

        td_error = torch.abs(Q - Q_target).detach()
        loss = torch.mean((Q - Q_target) ** 2 * weights.to(self.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

        return loss.item(), td_error
