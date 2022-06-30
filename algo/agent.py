import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.distributions as tdist

from hydra.utils import instantiate

class AtariAgent(nn.Module):
    def __init__(self, latent_size, hidden_size, depth, action_space=None,):
        super().__init__()
        layers = [nn.Linear(latent_size, hidden_size), nn.ReLU()]
        for i in range(depth):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, action_space.n))
        self.model = nn.Sequential(*layers)


    def forward(self, observation):
        return self.model(observation)

    def step(self, observation, deterministic=False):
        logits = self(observation)
        dist = tdist.OneHotCategoricalStraightThrough(logits=logits)
        action = dist.mode() if deterministic else dist.rsample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy

class Critic(nn.Module):
    def __init__(self, latent_size, hidden_size, depth, ):
        super().__init__()
        layers = [nn.Linear(latent_size, hidden_size), nn.ReLU()]
        for i in range(depth):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, 1))
        self.model = nn.Sequential(*layers)


    def forward(self, observation):
        return self.model(observation)
