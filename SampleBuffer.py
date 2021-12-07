from collections import deque
import numpy as np
import itertools

class SampleBuffer():
    def __init__(self, buffer_size=15000, shape=(84,84), num_frames=4):
        """
        Initializing a SampleBuffer for expierence Replay
        :param buffer_size: Determines the Buffer size which is choosen
        """
        self.size = buffer_size
        self.current = 0  # index to write to

        self.states = np.empty((self.size, shape[0], shape[1], num_frames), dtype=np.float32)
        self.actions = np.empty(self.size, dtype=np.int32)
        self.next_rewards = np.empty(self.size, dtype=np.float32)
        self.next_states = np.empty((self.size, shape[0], shape[1], num_frames), dtype=np.float32)
        self.dones = np.empty(self.size, dtype=np.int32)
        #self.states = deque(maxlen=buffer_size)
        #self.next_states = deque(maxlen=buffer_size)
        #self.actions = deque(maxlen=buffer_size)
        #self.next_rewards = deque(maxlen=buffer_size)
        #self.dones = deque(maxlen=buffer_size)


    def add_experience(self, state,  actions, next_reward, next_states, done):
        """
        Adds an Experience to the buffer
        :param state: State at time t
        :param actions: Action at time t
        :param next_reward: Reward at time t+1
        :param next_states: Sate at time t+1
        :param done: Done Flag at time t+1
        """
        self.states[self.current, ...] = state
        self.actions[self.current] = actions
        self.next_rewards[self.current] = next_reward
        self.next_states[self.current, ...] = next_states
        self.dones[self.current] = done
        self.current = (self.current + 1) % self.size


    def get_batch(self, batchsize = 50):
        """
        Get a Batch of the Sample Buffer to Train on
        :param batchsize: Number of Samples to be returned
        :return: Batch of Samples
        """
        index = np.random.randint(batchsize, len(self.states))
        lower_limit = index-batchsize
        states = self.states[index-batchsize:index, :, :, :]
        next_states = self.next_states[index-batchsize:index, :, :, :]
        actions = self.actions[index-batchsize:index]
        next_rewards = self.next_rewards[index-batchsize:index]
        dones = self.dones[index-batchsize:index]

        return states, next_states, actions, next_rewards, dones

    def get_buffer_length(self):
        """
        Gives Back the Current length of the SampleBuffer
        :return: Lenght of SampleBuffer
        """
        return len(self.states)

