from modules.diagnostics import Logger

class ConfigurationProvider(object):

    @staticmethod
    def CreateConfiguration():
        configuration = Configuration()
        configuration.input_text_file_path = "input_data\\shakespeare.txt"
        configuration.number_of_epochs = 2500
        configuration.batch_size = 10000
        configuration.buffer_size = 10000
        configuration.embedding_dim = 110
        configuration.units = 600
        configuration.tensorboard_logs_dir = 'logs\\tensorboard'
        configuration.checkpoint_dir = 'logs\\training_checkpoints'
        configuration.early_stop_if_no_improvement_epoch_limit = 0

        # https://www.tensorflow.org/tensorboard/migrate
        # https://www.tensorflow.org/tutorials/text/text_generation
        # Start tensorboard server https://github.com/tensorflow/tensorboard/blob/master/README.md
        # https://www.tensorflow.org/tensorboard/r1/graphs
        # https://www.tensorflow.org/tensorboard/r1/summaries

        Logger.Log('Configuration created')
        return configuration

class Configuration(object):

    def __init__(self):
        self.input_text_file_path = 'n/a'
        self.number_of_epochs = -1
        self.batch_size = -1
        self.buffer_size = -1
        self.embedding_dim = -1
        self.units = -1
        self.tensorboard_logs_dir = ''
        self.checkpoint_dir = ''
        self.early_stop_if_no_improvement_epoch_limit = None