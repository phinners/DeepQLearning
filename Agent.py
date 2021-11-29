
import gym
from DeepQModel import DeepQModel
from SampleBuffer import SampleBuffer
import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion
import datetime
from DuelingDeepQModel import DuelingDeepQModel
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class Agent():
    def __init__(self, Environment, Load_Weights=False, Weights_Path=""):
        """
        Initializing the DoubleQ-Learning Agent
        :param Environment: Name of the OpenAi Gym Environment
        :param Load_Weights: Flag to determine, if recent saved weights should be loaded
        :param Weights_Path: Path to the Weights which should be loaded
        """
        self.env_name = Environment
        self.env = gym.make(self.env_name)          # Create OpenAI Gym environment with env_name
        self.env.reset()

        self.action_size = self.env.action_space.n  # Number of Outputs from the Neural Network
        self.shape_frame = [84, 84]                 # Shape of Input to the Neural Network
        self.num_frames = 4                         # Number of Stacked Frames for Input to Neural Network

        self.epsilon = 0.9                          # Propability of random action choice in epsilon greedy
        self.epsilon_decrease = 0.005/100            # Decrease of Epsilon per Episode
        self.epsilon_min = 0.05                     # Lower Bound of Epsilon Value

        self.gamma = 0.9

        self.EPISODES = 20000                       # Number of Episodes to play
        self.episode_counter = 0                    # Counter of Episodes

        self.scores = []                            # Buffer of Scores for Visualizing the Performance
        self.episodes = []                          # Buffer of Episodes for Visualizing the Performance

        self.starttime_episode = datetime.datetime.now() # Starttime of Episode

        self.stepcounter = 1                        # Counter of taken Steps

        self.Weights_Path = Weights_Path            # Path where Weights are stored
        # Initialization of target Model
        self.target_model = DeepQModel(self.num_frames,
                                       self.shape_frame,
                                       self.action_size)
        self.online_model = DeepQModel(self.num_frames,
                                       self.shape_frame,
                                       self.action_size)

        optimizer = Adam(learning_rate=1e-3)
        self.target_model.compile(optimizer, loss=tf.keras.losses.Huber())
        self.online_model.compile(optimizer, loss=tf.keras.losses.Huber())

        self.target_model.build((None, self.shape_frame[0], self.shape_frame[1], self.num_frames))
        self.target_model.summary()
        self.online_model.build((None, self.shape_frame[0], self.shape_frame[1], self.num_frames))
        #Set Weights of online Model to the same as target Model
        self.online_model.set_weights(self.target_model.get_weights())
        self.online_model.summary()

        #Load weights of recent saved Weights
        if Load_Weights:
            print("Load Recent Weights")
            self.target_model.load_weights(self.Weights_Path)
            self.online_model.load_weights(self.Weights_Path)

        #Initialize Sample Buffer (Experience Replay)
        self.sample_buffer = SampleBuffer()
        #Initialize state Buffer, for stacking the number of frames for input into the Model
        self.state_buffer = deque(maxlen=self.num_frames + 1)
        #Initialize reward Buffer for Calculation purpose
        self.reward_buffer = deque(maxlen=2)

        self.figurename = "Results" #plt.figure("Results")

    def resize_observation(self, observation):
        """
        Resize the return of the environment to the predeterined size
        :param observation: Return of the environment after taking an action
        :return: Gives the resized observation back
        """
        resized = cv2.resize(observation, (84, 84))
        grayscale = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        binary = cv2.threshold(grayscale, 140, 255, cv2.THRESH_BINARY)[1]

        #resized = cv2.resize(grayscale, (84, 84))
        return binary

    def policy(self, epsilon, state):
        """
        Function for determining the action to take in the current stat
        :param epsilon: Possibility of random action choice (epsilon greedy)
        :param state: Input into the Model for action choice
        :return: Gives back the action to take
        """
        if np.random.rand() < epsilon:
            #print("Random")
            return np.random.randint(self.action_size)
        else:
            #print("Not Random")
            # print("Q Value Action")
            # obtain the current best Q values from the DeepQ network
            array = np.moveaxis(np.asarray(state), [0], [2])
            array = array.reshape(-1, array.shape[0], array.shape[1], array.shape[2])
            tensor = tf.convert_to_tensor(array, dtype=tf.float32)
            Q_values = self.online_model.predict(tensor)
            # take the action that results in the highest Q value
            #print(Q_values)
            return np.argmax(Q_values[0])

    def play_episode(self):
        """
        Function to play one episode on the environment
        """
        score = 0
        self.starttime_episode = datetime.datetime.now().replace(microsecond=0)
        # Reset environment to begin the episode at a random position
        self.env.reset()
        self.make_initial_steps(22)
        while(1):
            if len(self.state_buffer) != 4:
                action = self.env.action_space.sample()
            else:
                #Determin Action to Take
                action = self.policy(self.epsilon, self.state_buffer)

            # Play the Action on the environment and get the return values
            observation, reward, done, info = self.env.step(action)
            # Resize Observation
            resized = self.resize_observation(observation)
            self.state_buffer.append(resized)
            self.reward_buffer.append(reward)
            self.stepcounter += 1
            score += reward

            #plt.figure()
            #plt.imshow(resized)
            #plt.show()

            # Add Values to the Sample Buffer
            if len(self.state_buffer) >= 5:
                state = np.moveaxis(np.asarray(self.state_buffer), [0], [2])
                state = state.reshape(-1, state.shape[0], state.shape[1], state.shape[2])
                self.sample_buffer.add_experience(state[:, :, :, :4],       # Add State(t)
                                                  state[:, :, :, 1:5],      # Add State(t+1)
                                                  action,                   # Add action(t)
                                                  self.reward_buffer[1],    # Add reward(t+1)
                                                  done)                     # Done Flag

            # If we have enough entries in the sample buffer --> learn
            if self.sample_buffer.get_buffer_length() > 50:
                self.learn()

            # Save Weihghts to storage and update target network periodically
            if (self.stepcounter % 25000) == 0:
                self.save_current_weights()

            # If the episode is finished, recalculate epsilon and add score to score buffer
            if done:
                self.epsilon = self.epsilon - self.epsilon_decrease
                if self.epsilon < self.epsilon_min:
                    self.epsilon = self.epsilon_min
                self.scores.append(score)
                #self.learn()
                break

            if reward != 0.0:
                self.make_initial_steps(22)

    def learn(self):
        """
        Learn from the action´s we´ve taken so far
        """
        # Get a Batch of the sample buffer
        states, next_states, actions, next_rewards, dones = self.sample_buffer.get_batch()
        # Calculate the QValues for the next_states from online model and target model
        QValues = self.online_model.predict(tf.convert_to_tensor(np.array(next_states), dtype=tf.float32))
        next_QValues = self.target_model.predict(tf.convert_to_tensor(np.array(next_states), dtype=tf.float32))
        # Create Copy of QValues where the reccalculated QValues are stored
        updated_QValues = QValues.copy()

        #Iterate through every state
        for i in range(len(QValues)):
            # Update Formula: updated_QValue_online[action] = Reward(t+1) + gamma * QValue_target(t+1, action)
            # action --> action of highest QValue of online Model in State(t+1)
            next_reward = next_rewards[i]
            # Determine Action to take of state(t+1), highest QValue of online Model at State (t+1)
            action = np.where(QValues[i] == np.amax(QValues[i]))[0][0]
            updated_QValues[i][action] = (self.gamma * next_QValues[i][action]) + next_reward
            #print("Old QValue: " + str(QValues[i][action]) + " QValue: " + str(updated_QValues[i][action]) + " Reward: " + str(next_reward) + " Action: " + str(action))
            pass

        # Just for debbuging
        if (self.stepcounter % 5000) == 0:
            print("Current QValues at stepcounter " + str(self.stepcounter) + ": " + str(updated_QValues[-1]))

        # Fit the Model with the updated_QValues and the States(t)
        self.online_model.fit(x=tf.convert_to_tensor(np.array(states), dtype=tf.float32),
                              y=updated_QValues,
                              verbose=0)

    def play(self):
        """
        Main Function of the Agent, Iterates the number of Actions times and outputs the current performance
        """
        # Iterate Through Episodes
        for i in range(self.EPISODES):
            # Play Episode
            self.play_episode()
            time_used = datetime.datetime.now().replace(microsecond=0) - self.starttime_episode
            # Print Out the reached performance
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

            # Create Image of Performance
            plt.figure(self.figurename)
            plt.plot(self.episodes, self.scores)
            plt.savefig('Current_Performance.png')

        #After finishing the Training, save the Weights to Storage
        self.save_current_weights()
        print(str(datetime.datetime.now().replace(microsecond=0)) + "Finished Training!")

    def save_current_weights(self):
        print(str(datetime.datetime.now().replace(microsecond=0)) + "Update Target Model and saved current weights!")
        self.target_model.set_weights(self.online_model.get_weights())
        self.online_model.save_weights(self.Weights_Path)

    def make_initial_steps(self, number = 20):
        for i in range(number):
            action = self.env.action_space.sample()
            self.env.step(action)
            #print("Make Initial Steps: " + str(i))

if __name__ == '__main__':
    Agent = Agent(Environment="Pong-v0",
                  Load_Weights=False,
                  Weights_Path="recent_weights.hdf5")
    Agent.play()

