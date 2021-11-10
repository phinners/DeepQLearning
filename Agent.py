
import gym
from DeepQModel import DeepQModel
from SampleBuffer import SampleBuffer
import numpy as np
import cv2
from collections import deque

class Agent():
    def __init__(self, Environment):
        self.env_name = Environment
        self.env = gym.make(self.env_name)
        self.env.reset()

        self.action_size = self.env.action_space.n
        self.shape_frame = [84, 84] #self.env.observation_space.shape
        self.num_frames = 4

        self.EPISODES = 20000
        self.episode_counter = 0
        self.max_average = -21.0

        self.scores = []
        self.episodes = []
        self.average = []

        self.target_model = DeepQModel(self.num_frames, self.shape_frame, self.action_size)
        self.online_model = DeepQModel(self.num_frames, self.shape_frame, self.action_size)

        self.sample_buffer = SampleBuffer()
        self.state_buffer = deque(maxlen = self.num_frames)

    def resize_observation(self, observation):
        grayscale = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(grayscale, (84,84))
        return resized

    def policy(self, epsilon, state):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        else:
            # print("Q Value Action")
            # obtain the current best Q values from the DeepQ network
            Q_values = self.online_model.model.predict(state[np.newaxis])
            # take the action that results in the highest Q value
            return np.argmax(Q_values[0])

    def play_episode(self):
        while(1):
            if len(self.state_buffer) != 4:
                action = self.env.action_space.sample()
            else:
                action = self.policy(state, model, epsilon)

            observation, reward, done, info = self.env.step(action)
            resized = self.resize_observation(observation)
            self.state_buffer.append(resized)

if __name__ == '__main__':
    Agent("Pong-v0")