from collections import deque
import numpy as np
import itertools

class SampleBuffer():
    def __init__(self, buffer_size=5000):
        self.states = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)

    def add_experience(self, state, actions, reward, done):
        self.states.append(state)
        self.actions.append(actions)
        self.rewards.append(reward)
        self.dones.append(done)

    def get_batch(self, batchsize = 50):
        index = np.random.randint(batchsize, len(self.states))
        lower_limit = index-batchsize
        #list(itertools.islice(q, 3, 7))
        #states = self.states[index-batchsize:index]
        states = list(itertools.islice(self.states, index-batchsize, index))
        actions = list(itertools.islice(self.actions, index - batchsize, index))
        rewards = list(itertools.islice(self.rewards, index - batchsize, index))
        dones = list(itertools.islice(self.dones, index - batchsize, index))
        #actions = self.actions[index - batchsize:index]
        #rewards = self.rewards[index - batchsize:index]
        #dones = self.dones[index - batchsize:index]

        return states, actions, rewards, dones

    def get_buffer_length(self):
        return len(self.states)

