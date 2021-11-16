
import gym
from DeepQModel import DeepQModel
from SampleBuffer import SampleBuffer
import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion
import datetime

class Agent():
    def __init__(self, Environment, Load_Weights=False, Weights_Path=""):
        self.env_name = Environment
        self.env = gym.make(self.env_name)
        self.env.reset()

        self.action_size = self.env.action_space.n
        self.shape_frame = [84, 84] #self.env.observation_space.shape
        self.num_frames = 4

        self.epsilon = 0.9
        self.epsilon_decrease = 0.1/100
        self.epsilon_min = 0.05

        self.EPISODES = 20000
        self.episode_counter = 0
        self.max_average = -21.0

        self.scores = []
        self.episodes = []
        self.average = []

        self.starttime = datetime.datetime.now()
        self.starttime_episode = datetime.datetime.now()

        self.stepcounter = 1

        self.Weights_Path = Weights_Path
        self.target_model = DeepQModel(self.num_frames, self.shape_frame, self.action_size)
        self.online_model = DeepQModel(self.num_frames, self.shape_frame, self.action_size)

        if Load_Weights:
            print("Load Recent Weights")
            self.target_model.model.load_weights(self.Weights_Path)
            self.online_model.model.load_weights(self.Weights_Path)

        self.sample_buffer = SampleBuffer()
        self.state_buffer = deque(maxlen=self.num_frames)

        self.figurename = "Results" #plt.figure("Results")

    def resize_observation(self, observation):
        grayscale = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(grayscale, (84, 84))
        return resized

    def policy(self, epsilon, state):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)
        else:
            # print("Q Value Action")
            # obtain the current best Q values from the DeepQ network
            array = np.moveaxis(np.asarray(state), [0], [2])
            array = array.reshape(-1, array.shape[0], array.shape[1], array.shape[2])
            Q_values = self.online_model.model.predict(array)
            # take the action that results in the highest Q value
            return np.argmax(Q_values[0])

    def play_episode(self):
        score = 0
        self.starttime_episode = datetime.datetime.now().replace(microsecond=0)
        self.env.reset()
        while(1):
            if len(self.state_buffer) != 4:
                action = self.env.action_space.sample()
            else:
                action = self.policy(self.epsilon, self.state_buffer)

            observation, reward, done, info = self.env.step(action)
            resized = self.resize_observation(observation)
            self.state_buffer.append(resized)

            if len(self.state_buffer) >= 4:
                state = np.moveaxis(np.asarray(self.state_buffer), [0], [2])
                state = state.reshape(-1, state.shape[0], state.shape[1], state.shape[2])
                self.sample_buffer.add_experience(state, action, reward, done)


            if self.sample_buffer.get_buffer_length() > 100:
                #print("Learn: " + str(self.stepcounter))
                self.stepcounter += 1
                self.learn()

            if (self.stepcounter % 5000) == 0:
                print("Update Target Model and saved current weights!")
                self.target_model.model.set_weights(self.online_model.model.get_weights())
                self.online_model.model.save_weights(self.Weights_Path)

            score += reward
            if done:
                self.epsilon = self.epsilon - self.epsilon_decrease
                if self.epsilon < self.epsilon_min:
                    self.epsilon = self.epsilon_min
                self.scores.append(score)
                break

    def learn(self):
        states, actions, rewards, dones = self.sample_buffer.get_batch()
        QValues = self.target_model.model.predict(np.array(states))
        updated_QValues = QValues

        for i in range(len(QValues)):
            updated_QValues[i][actions[i]] = 0.99 * QValues[i][actions[i]] + rewards[i]

        self.online_model.model.fit(x=np.array(states),
                                    y=updated_QValues,
                                    verbose=0)

    def play(self):
        for i in range(1000):
            self.play_episode()
            time_used = datetime.datetime.now().replace(microsecond=0) - self.starttime_episode
            print(str(datetime.datetime.now().replace(microsecond=0)) +
                  "  Finished Episode: " +
                  str(i) +
                  " in " +
                  str(time_used) +
                  " Score: " +
                  str(self.scores[-1]) +
                  " Epsilon: " +
                  str(self.epsilon))
            self.episodes.append(i)
            plt.figure(self.figurename)
            plt.plot(self.episodes, self.scores)
            plt.savefig('Current_Performance.png')
            if (i % 100) == 0 and i != 0:
                plt.figure()
                plt.plot(self.episodes,self.scores)

if __name__ == '__main__':
    Agent = Agent(Environment="Pong-v0",
                  Load_Weights=True,
                  Weights_Path="recent_weights.hdf5")
    Agent.play()

