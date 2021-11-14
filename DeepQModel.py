# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input

class DeepQModel(tf.keras.Model):
    def __init__(self, num_frames, shape_frame, num_actions):
        super(DeepQModel, self).__init__()

        self.num_frames = num_frames
        self.shape_frame = shape_frame
        print(self.shape_frame[0])
        self.num_actions = num_actions

        self.model = Sequential()

        self.model.add(Input(shape=(shape_frame[0], shape_frame[1], num_frames)))
        self.model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=4, data_format="channels_last", activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        self.model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=2, data_format="channels_last", activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, data_format="channels_last", activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))

        self.model.add(Flatten())

        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(num_actions, activation='linear'))

        self.model.summary()

    def load_model(self, file):
        self.model = load_model(file)
