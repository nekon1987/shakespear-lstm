import time, sys, os
from datetime import datetime
import tensorflow as tf
import numpy as np
from modules.configuration import Configuration
from modules.tensorboard import TensorboardManager

class TrainingSupervisor():

    def __init__(self, configuration: Configuration, tensorboardManager: TensorboardManager):
        self.configuration = configuration
        self.tensorboardManager = tensorboardManager
        self.checkpoint = None
        self.optimizer = tf.keras.optimizers.Adam()
        self.temp_lstm_hidden_state = None
        self.network_not_improving_epochs_counter = 0
        self.network_lowest_loss_value = sys.maxsize

    def should_early_stop_epoch_update(self, current_loss):
        if(current_loss.numpy() > self.network_lowest_loss_value):
            self.network_not_improving_epochs_counter += 1
        else:
            self.network_lowest_loss_value = current_loss.numpy()
            self.network_not_improving_epochs_counter = 0

        print('Lower Loss {:.4f} Not improved for: {} epochs'.format(self.network_lowest_loss_value, self.network_not_improving_epochs_counter))
        return self.network_not_improving_epochs_counter >= self.configuration.early_stop_if_no_improvement_epoch_limit

    def loss_function(self, labels, logits):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name = 'Loss Function')

    def configure_checkpoints(self, model):
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=model)
        self.checkpoint_directory = "{}\Shakespear-{}".format(self.configuration.checkpoint_dir,
                                                              datetime.now().strftime("%Y%m%d%H%M%S"))

    def restore_latest(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_directory))

    def save_checkpoint(self, suffix):
        self.checkpoint.save(file_prefix=os.path.join(self.checkpoint_directory, "ckpt_{}".format(suffix)))

    def train_model(self, model, dataset):
        writer = self.tensorboardManager.create_logWritter()
        with writer.as_default():
            for epoch in range(self.configuration.number_of_epochs):
                start = time.time()
                model.reset_states()
                for (batch, (input, target)) in enumerate(dataset):
                    loss = self.step(model, input, target)

                mean_loss = tf.math.reduce_mean(loss)
                tf.summary.scalar('mean batch loss', mean_loss, step=epoch)
                print('Epoch {} Took: {:.2f} Loss {:.4f}'.format(epoch + 1, time.time() - start, mean_loss))

                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint("ckpt_{epoch}")

                if self.should_early_stop_epoch_update(mean_loss):
                    self.save_model(model)
                    return

    #@tf.function
    def step(self, model, input, target):
        self.tensorboardManager.start_tensorboard_graph_trace()

        with tf.GradientTape() as tape:

            predictions, self.temp_lstm_hidden_state = model(input, self.temp_lstm_hidden_state)

            target = tf.reshape(target, (-1,))
            loss = self.loss_function(target, predictions)
            grads = tape.gradient(loss, model.variables)
            self.optimizer.apply_gradients(zip(grads, model.variables))
            self.tensorboardManager.commit_tensorboard_graph_trace('{}-graph-uno'.format(datetime.now()))

            self.log_summaries(model, grads, loss, predictions, self.temp_lstm_hidden_state)

            return loss

    def save_model(self, model):
        test_descriptive_label = datetime.now().strftime(
            "%Y%m%d%H%M%S") + '-' + 'shapespear-lstm-' + str(self.network_lowest_loss_value)
        results_directory = os.getcwd() + '\\' + 'output_model\\'
        os.mkdir(results_directory + test_descriptive_label)
        model_file_path = results_directory + test_descriptive_label + '\\' + test_descriptive_label + '.weights'
        model.save_weights(model_file_path)
        print('Saved model: ' + test_descriptive_label)

    def log_summaries(self, model, grads, loss, predictions, hidden_state):
        return
        tf.summary.histogram(name='Weights-Embedding', data=model.layers[0].trainable_variables[0], step=self.optimizer.iterations)
        tf.summary.histogram(name='Weights-LSTM-Input', data=model.layers[1].trainable_variables[0], step=self.optimizer.iterations)
        tf.summary.histogram(name='Weights-LSTM-Hidden', data=model.layers[1].trainable_variables[1], step=self.optimizer.iterations)
        tf.summary.histogram(name='Weights-LSTM-Output', data=model.layers[1].trainable_variables[2], step=self.optimizer.iterations)
        tf.summary.histogram(name='Weights-Dense-Input', data=model.layers[2].trainable_variables[0], step=self.optimizer.iterations)
        tf.summary.histogram(name='Weights-Dense-Output', data=model.layers[2].trainable_variables[1], step=self.optimizer.iterations)

        tf.summary.histogram(name='Gradients-Embedding', data=grads[0], step=self.optimizer.iterations)
        tf.summary.histogram(name='Gradients-LSTM-Input', data=grads[1], step=self.optimizer.iterations)
        tf.summary.histogram(name='Gradients-LSTM-Hidden', data=grads[2], step=self.optimizer.iterations)
        tf.summary.histogram(name='Gradients-LSTM-Output', data=grads[3], step=self.optimizer.iterations)
        tf.summary.histogram(name='Gradients-Dense-Input', data=grads[4], step=self.optimizer.iterations)
        tf.summary.histogram(name='Gradients-Dense-Output', data=grads[5], step=self.optimizer.iterations)

        tf.summary.histogram(name='LSTM-Hidden-Input', data=hidden_state[0], step=self.optimizer.iterations)
        tf.summary.histogram(name='LSTM-Hidden-Output', data=hidden_state[1], step=self.optimizer.iterations)

        tf.summary.scalar('step loss mean', tf.math.reduce_mean(loss), step=self.optimizer.iterations)
        tf.summary.scalar('step loss std', tf.math.reduce_std(loss), step=self.optimizer.iterations)
        tf.summary.scalar('step loss variance', tf.math.reduce_variance(loss), step=self.optimizer.iterations)