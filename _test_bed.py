import tensorflow as tf
from modules.infrastructure import InfrastructureService
from modules.configuration import ConfigurationProvider

cfg = ConfigurationProvider.CreateConfiguration()
infrastructureService = InfrastructureService(cfg)
infrastructureService.SetupTensorflowForGpuMode()
infrastructureService.SetupTensorboard()

tf.summary.trace_on(graph=True, profiler=True)
# *****************************************************************************************

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

(train_images, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0

def custom_callback():
    writer = tf.summary.create_file_writer(cfg.tensorboard_logs_dir)
    with writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=cfg.tensorboard_trace_profiler_logs_dir)
        writer.flush()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=cfg.tensorboard_logs_dir)

# Train the model.
model.fit(
    train_images,
    train_labels,
    batch_size=64,
    epochs=5)

custom_callback()
# *****************************************************************************************
print('done')