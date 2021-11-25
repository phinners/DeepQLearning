# TensorFlow
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D

class DuelingDeepQModel(tf.keras.Model):
    def __init__(self, num_frames, shape_frame, num_actions):
        """
        Initializing a Dueling Deep Q model
        :param num_frames: currently not used
        :param shape_frame: currently not used
        :param num_actions: Number of Outputs of the Model
        """
        super(DuelingDeepQModel, self).__init__()

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
        self.Flatten = Flatten()

        #Layers for Value and Advantage Calculation
        self.Dense_Value_1 = Dense(512, activation='relu')
        self.Dense_Value_2 = Dense(1, activation='relu')

        self.Dense_Advantage_1 = Dense(512, activation='relu')
        self.Dense_Advantage_2 = Dense(num_actions, activation='relu')

    def call(self, input_data):
        """
        Function which is called when predict is called
        :param input_data: Input Date for the Model
        :return: Gives Back the QValues
        """
        x = self.Conv2D_1(input_data)
        x = self.Conv2D_2(x)
        x = self.Conv2D_3(x)
        x = self.Flatten(x)

        Values = self.Dense_Value_1(x)
        Values = self.Dense_Value_2(Values)

        Advantage = self.Dense_Advantage_1(x)
        Advantage = self.Dense_Advantage_2(Advantage)

        #Calculate QValues from Values and Advantage
        QValues = Values + (Advantage - tf.math.reduce_mean(Advantage, axis=1, keepdims=True))
        return QValues

