from modules.diagnostics import Logger

class ConfigurationProvider(object):

    @staticmethod
    def CreateConfiguration():
        configuration = Configuration()
        configuration.input_text_file_path = "input_data\\shakespeare.txt"
        configuration.number_of_epochs = 40
        configuration.batch_size = 5000
        configuration.buffer_size = 20000
        configuration.embedding_dim = 100
        configuration.units = 512
        configuration.tensorboard_logs_dir = 'logs\\tensorboard'
        configuration.tensorboard_trace_profiler_logs_dir = 'logs\\debug_traces'
        configuration.checkpoint_dir = './training_checkpoints_1'
        configuration.profiler_output_logs_dir = 'logs\\profiler_output'

        # https://www.tensorflow.org/tensorboard/migrate
        # https://www.tensorflow.org/tutorials/text/text_generation
        # Start tensorboard server https://github.com/tensorflow/tensorboard/blob/master/README.md
        # https://www.tensorflow.org/tensorboard/r1/graphs
        #  https://www.tensorflow.org/tensorboard/r1/summaries

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
        self.tensorboard_trace_profiler_logs_dir = ''
        self.checkpoint_dir = ''