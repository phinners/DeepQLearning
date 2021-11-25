# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam

class DeepQModel(tf.keras.Model):
    def __init__(self, num_frames, shape_frame, num_actions):
        """
        Initializing a Deep Q-Model
        :param num_frames: currently not used
        :param shape_frame: currently not used
        :param num_actions: number of output nodes of the model
        """
        super(DeepQModel, self).__init__()

        # Add Some Convulutional Layers for Image Recognition
        self.Conv2D_1 = Conv2D(filters=32,
                              kernel_size=(8, 8),
                              strides=4,
                              data_format="channels_last",
                              activation='relu',
                              kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2))
        self.Conv2D_2 = Conv2D(filters=64,
                               kernel_size=(4, 4),
                               strides=2,
                               data_format="channels_last",
                               activation='relu',
                               kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2))
        self.Conv2D_3 = Conv2D(filters=64,
                               kernel_size=(3, 3),
                               strides=1,
                               data_format="channels_last",
                               activation='relu',
                               kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2))
        self.flatten = Flatten()

        self.Dense_1 = Dense(512, activation='relu')
        self.Dense_2 = Dense(num_actions, activation='linear')

    def call(self, input_data):
        """
        Function which is called when predict is called
        :param input_data: Input Date for the Model
        :return: Gives Back the QValues
        """
        x = self.Conv2D_1(input_data)
        x = self.Conv2D_2(x)
        x = self.Conv2D_3(x)
        x = self.flatten(x)
        x = self.Dense_1(x)
        QValues = self.Dense_2(x)

        return QValues
