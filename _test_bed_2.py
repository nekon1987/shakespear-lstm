import numpy as np
import tensorflow as tf
from tensorboard import default
from tensorboard import program

tb = program.TensorBoard(default.get_plugins())
tb.configure(argv=[None, '--logdir', "./logdir"])
url = tb.launch()

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.dense1 = tf.keras.layers.Dense(100)

    def call(self, inputs):
        outputs = self.dense1(inputs)
        return outputs


model = Model()
optimizer = tf.keras.optimizers.Adam()


@tf.function
def train(data):
    with tf.name_scope("xxx"):
        with tf.GradientTape() as tape:
            y = model(data)
            loss = tf.reduce_mean(tf.square(y))
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))


x = np.random.rand(10, 100).astype(np.float32)
# y = model(x)
# train.python_function(x)
tf.summary.trace_on()
train(x)
writer = tf.summary.create_file_writer("./logdir")
with writer.as_default():
    tf.summary.trace_export("graph2", step=0)
    tf.summary.trace_off()
    writer.flush()

print('done')