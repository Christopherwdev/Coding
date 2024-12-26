import os
import torch
import torch.nn as nn
import numpy as np
from networks import *
from replay_buffer import *
from config import *
import wandb

# ... (DDPGAgent class remains largely the same, but with improvements)

    def get_action(self, observation: np.ndarray, evaluation=False):
        #Improved handling of single observations
        observation = np.array(observation, dtype=np.float32)
        if observation.ndim == 1:
            observation = np.expand_dims(observation, axis=0)
        state = torch.tensor(observation, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            if not evaluation:
                actions = self.actor.noisy_forward(state).detach().cpu().numpy()
            else:
                actions = self.actor.forward(state).detach().cpu().numpy()
        actions = np.clip(actions, self.lower_bound, self.upper_bound)
        return actions[0]


    def learn(self, batch_size=BATCH_SIZE):
        if self.replay_buffer.check_buffer_size(batch_size) is False:
            return 0.0, 0.0, 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # ... (rest of the learn function remains largely the same)
