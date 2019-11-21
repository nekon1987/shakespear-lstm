import tensorflow as tf

from modules.diagnostics import Logger
from modules.configuration import ConfigurationProvider
from training_supervisor import TrainingSupervisor

class ShakespeareModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size):
        super(ShakespeareModel, self).__init__()
        self.units = units
        self.batch_size = batch_size

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=5)

        self.lstm = tf.keras.layers.LSTM(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_activation='sigmoid',
                                       recurrent_initializer='glorot_uniform')

        self.fc = tf.keras.layers.Dense(vocab_size)
        self.previous_states = None


    @tf.function
    def call(self, inputs):
        Logger.Log("input shape: {}".format(inputs.shape))
        inputs = self.embedding(inputs)
        Logger.Log("embedding shape: {}".format(inputs.shape))
        result = self.lstm(inputs, self.previous_states)
        output = result[0]
        Logger.Log("embedding shape: {}".format(output.shape))

        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        Logger.Log("embedding shape: {}".format(x.shape))


        #previous_states = result[1:]
        #self.previous_states = previous_states
        # TODO - update LSTM !!!

        return x