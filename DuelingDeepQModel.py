# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam

class DuelingDeepQModel(tf.keras.Model):
    def __init__(self, num_frames, shape_frame, num_actions):
        super(DuelingDeepQModel, self).__init__()

        self.Input = Input(shape=(shape_frame[0], shape_frame[1], num_frames))
        self.Conv2D_1 = Conv2D(filters=32, kernel_size=(8, 8), strides=4, data_format="channels_last", activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2))
        self.Conv2D_2 = Conv2D(filters=64, kernel_size=(4, 4), strides=2, data_format="channels_last", activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2))
        self.Conv2D_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, data_format="channels_last", activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2))


        self.Dense_Value = Dense(1, activation='relu')
        self.Dense_Advantage = Dense(num_actions, activation='relu')

        optimizer = Adam(1e-3)
        self.model.compile(optimizer, loss=tf.keras.losses.Huber())

        self.model.summary()

    def call(self, input_data):
        x = self.Input(input_data)
        x = self.Conv2D_1(x)
        x = self.Conv2D_2(x)
        x = self.Conv2D_3(x)
        Values = self.Dense_Value(x)
        Advantage = self.Dense_Advantage(x)

        QValues = Values + (Advantage - tf.math.reduce_mean(Advantage, axis=1, keepdims=True))
        return QValues

