from modules.configuration import Configuration
import tensorflow as tf

class TensorboardManager():

    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.graph_stored = False


    def start_tensorboard_graph_trace(self):
        if not self.graph_stored:
            tf.summary.trace_on(graph=True, profiler=True)

    def commit_tensorboard_graph_trace(self, name):
        if not self.graph_stored:
            writer = tf.summary.create_file_writer(self.configuration.tensorboard_logs_dir)
            with writer.as_default():
                tf.summary.trace_export(
                    name=name,
                    step=0,
                    profiler_outdir=self.configuration.tensorboard_trace_profiler_logs_dir)
                writer.flush()
                self.graph_stored = True