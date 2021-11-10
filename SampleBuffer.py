from collections import deque
import numpy as np

class SampleBuffer():
    def __init__(buffer_size=5000):
        states = deque(maxlen = buffer_size)
        actions = deque(maxlen = buffer_size)
        rewards = deque(maxlen = buffer_size)
        dones = deque(maxlen=buffer_size)

    def add_experience(self, state, actions, reward, done):
        self.states.append(state)
        self.actions.append(state)
        self.rewards.append(state)
        self.dones.append(state)

    def get_batch(self, batchsize = 50):
        index = np.random.randint(batchsize, len(self.states))
        states = self.states[index-batchsize:index]
        actions = self.actions[index - batchsize:index]
        rewards = self.rewards[index - batchsize:index]
        dones = self.dones[index - batchsize:index]

        return states, actions, rewards, dones

    def get_buffer_length(self):
        return len(self.states)

