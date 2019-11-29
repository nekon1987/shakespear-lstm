from modules.configuration import Configuration
from tensorboard import default
from tensorboard import program
import tensorflow as tf
import datetime

class TensorboardManager():

    def __init__(self, configuration: Configuration):
        self.tensorboard_current_run_log_directory = "{}\Shakespear-{}".format(configuration.tensorboard_logs_dir,
                                                              datetime.datetime.now().strftime("%Y%m%d%H%M%S") )
        self.configuration = configuration
        self.graph_stored = False


    def SetupTensorboard(self):
        tb = program.TensorBoard(default.get_plugins())
        tb.configure(argv=[None, '--logdir', self.configuration.tensorboard_logs_dir])
        url = tb.launch()
        print('Tensorboard available @ {}'.format(url))

    def create_logWritter(self):
        return tf.summary.create_file_writer(self.tensorboard_current_run_log_directory, flush_millis=500)

    def start_tensorboard_graph_trace(self):
        if not self.graph_stored:
            tf.summary.trace_on(graph=True, profiler=False)

    def commit_tensorboard_graph_trace(self, name):
        if not self.graph_stored:
            tf.summary.trace_export(
                name=name,
                step=0,
                profiler_outdir=self.tensorboard_current_run_log_directory)
            self.graph_stored = True
