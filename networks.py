import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, obs_dim, actions_dim, upper_bound):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(obs_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, actions_dim)
        self.upper_bound = upper_bound
        self.noise = None

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x)) # tanh to keep actions within [-1, 1]
        return x * self.upper_bound # Scale to the action bounds

    def noisy_forward(self, x):
        if self.noise is None:
            self.noise = torch.randn_like(self.layer3.weight)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x) + self.noise) # Add noise to the output
        return x * self.upper_bound

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(obs_dim + act_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
