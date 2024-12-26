import numpy as np
import random

class ReplayBuffer():
    def __init__(self, env, max_size):
        self.max_size = max_size
        self.env = env
        self.buffer = []
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append([state, action, reward, next_state, done])
        else:
            self.buffer[self.ptr] = [state, action, reward, next_state, done]
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def check_buffer_size(self, batch_size=64):
        return len(self.buffer) > batch_size
