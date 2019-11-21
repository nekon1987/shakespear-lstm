import time
import os
import tensorflow as tf

from modules.configuration import Configuration

class TrainingSupervisor():

    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.checkpoint = None
        self.optimizer = tf.keras.optimizers.Adam()

    def loss_function(self, labels, logits):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

    def configure_checkpoints(self, model):
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=model)

    def restore_latest(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.configuration.checkpoint_dir))

    def start_tensorboard_session(self):
        tf.summary.trace_on(graph=True, profiler=True)

    def commit_tensorboard_session(self, name):
        writer = tf.summary.create_file_writer(self.configuration.tensorboard_logs_dir)
        with writer.as_default():
            tf.summary.trace_export(
                name=name,
                step=0,
                profiler_outdir=self.configuration.tensorboard_trace_profiler_logs_dir)
            writer.flush()

    def train_model(self, model, dataset):
        for epoch in range(self.configuration.number_of_epochs):
            start = time.time()
            model.reset_states()
            for (batch, (input, target)) in enumerate(dataset):

                self.start_tensorboard_session()
                loss = self.step(model, input, target)
                self.commit_tensorboard_session('graph-uno')

                if batch % 5 == 0:
                    print('Epoch {} Batch {} Took: {} Loss {:.4f}'.format(epoch + 1, batch, time.time()-start, loss[0]))

            if (epoch + 1) % 10 == 0:
                self.checkpoint.save(file_prefix=os.path.join(self.configuration.checkpoint_dir, "ckpt_{epoch}"))



    #@tf.function
    def step(self, model, input, target):
        with tf.name_scope("step_scope"):
            with tf.GradientTape() as tape:

                predictions = model(input)  # passing to itself? TODO
                target = tf.reshape(target, (-1,))
                loss = self.loss_function(target, predictions)

                grads = tape.gradient(loss, model.variables)
                self.optimizer.apply_gradients(zip(grads, model.variables))
                return loss