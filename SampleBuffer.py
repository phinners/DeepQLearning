from collections import deque
import numpy as np
import itertools

class SampleBuffer():
    def __init__(self, buffer_size=15000):
        """
        Initializing a SampleBuffer for expierence Replay
        :param buffer_size: Determines the Buffer size which is choosen
        """

        self.states = deque(maxlen=buffer_size)
        self.next_states = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.next_rewards = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)

    def add_experience(self, state, next_states, actions, next_reward, done):
        """
        Adds an Experience to the buffer
        :param state: Input of Model at time t
        :param next_states: Input of Model at time t+1
        :param actions: Action of Model at time t+1
        :param next_reward: Reward of Model at time t+1
        :param done: Done Flag at time t+1
        """
        self.states.append(state)
        self.next_states.append(next_states)
        self.actions.append(actions)
        self.next_rewards.append(next_reward)
        self.dones.append(done)

    def get_batch(self, batchsize = 50):
        """
        Get a Batch of the Sample Buffer to Train on
        :param batchsize: Number of Samples to be returned
        :return: Batch of Samples
        """
        index = np.random.randint(batchsize, len(self.states))
        lower_limit = index-batchsize
        states = list(itertools.islice(self.states, index-batchsize, index))
        next_states = list(itertools.islice(self.next_states, index - batchsize, index))
        actions = list(itertools.islice(self.actions, index - batchsize, index))
        next_rewards = list(itertools.islice(self.next_rewards, index - batchsize, index))
        dones = list(itertools.islice(self.dones, index - batchsize, index))

        return states, next_states, actions, next_rewards, dones

    def get_buffer_length(self):
        """
        Gives Back the Current length of the SampleBuffer
        :return: Lenght of SampleBuffer
        """
        return len(self.states)

