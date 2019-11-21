import tensorflow as tf
import unidecode
import numpy as np
from keras_preprocessing.text import Tokenizer
from modules.configuration import Configuration

class DatasetManager(object):

    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.tokenizer = None

    def CreateTrainingSet(self):
        text = unidecode.unidecode(open(self.configuration.input_text_file_path).read())
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([text])
        encoded = self.tokenizer.texts_to_sequences([text])[0]
        vocabulary_size = len(self.tokenizer.word_index) + 1

        X = []
        Y = []
        for i in range(0, len(encoded) - 2, 2):
            X.append(encoded[i])
            Y.append(encoded[i + 1])

        X = np.expand_dims(X, 1)
        Y = np.expand_dims(Y, 1)
        dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(self.configuration.buffer_size)
        dataset = dataset.batch(self.configuration.batch_size, drop_remainder=True)

        return vocabulary_size, dataset

    def WordToNumber(self, word):
        return [self.tokenizer.word_index[word]]

    def NumberToWord(self, numericalIndex):
        return [self.tokenizer.index_word[numericalIndex]]