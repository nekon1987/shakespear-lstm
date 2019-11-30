import tensorflow as tf
import numpy as np

from modules.diagnostics import Logger
from modules.configuration import ConfigurationProvider
from training_supervisor import TrainingSupervisor

class ShakespeareModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size):
        super(ShakespeareModel, self).__init__()
        self.units = units
        self.batch_size = batch_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=5, name='embedding-layer')

        self.lstm = tf.keras.layers.LSTM(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_activation='sigmoid',
                                       recurrent_initializer='glorot_uniform',
                                       name='lstm-layer')

        self.dropout = tf.keras.layers.Dropout(0.5)

        self.fc = tf.keras.layers.Dense(vocab_size, name='dense-layer')
        self.previous_states = None

    def print_diagnostics(self):
        def print_parameters(prefix, array):
            if len(array) > 0:
                print('{}) kernel: {} recu-kernel: {} bias: {}'.format(
                    prefix,
                    np.average(array[0].numpy()),
                    np.average(array[1].numpy()),
                    np.average(array[2].numpy())))

        print_parameters('TV', self.lstm.trainable_variables)
        print_parameters('TW', self.lstm.trainable_weights)
        print_parameters('V', self.lstm.variables)
        print_parameters('W', self.lstm.weights)

    @tf.function
    def call(self, inputs, previous):
        embeddedInput = self.embedding(inputs)

        lstmOutput = self.lstm(embeddedInput, previous)
        lstmPrediction = lstmOutput[0]

        lstmPrediction = tf.reshape(lstmPrediction, (-1, lstmPrediction.shape[2]))

        dropped = self.dropout(lstmPrediction)

        finalPrediction = self.fc(dropped)

        previous_states = lstmOutput[1:]

        return finalPrediction, previous_states