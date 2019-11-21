import tensorflow as tf

from modules.tensorboard import TensorboardManager
from shakespeare_model import ShakespeareModel
from modules.infrastructure import InfrastructureService
from modules.configuration import ConfigurationProvider
from modules.datasets import DatasetManager
from training_supervisor import TrainingSupervisor

cfg = ConfigurationProvider.CreateConfiguration()
tensorboardManager = TensorboardManager(cfg)
infrastructureService = InfrastructureService(cfg)
datasetManager = DatasetManager(cfg)
trainingSupervisor = TrainingSupervisor(cfg, tensorboardManager)

infrastructureService.SetupTensorflowForGpuMode()
infrastructureService.SetupTensorboard()
vocabulary_size, training_dataset = datasetManager.CreateTrainingSet()

model = ShakespeareModel(vocabulary_size, cfg.embedding_dim, cfg.units, cfg.batch_size)

#model.compile(tf.keras.optimizers.Adam(), tf.keras.losses.categorical_crossentropy, run_eagerly=True)
#model.build((None, 5))
#model.summary()

trainingSupervisor.configure_checkpoints(model)
trainingSupervisor.train_model(model, training_dataset)
trainingSupervisor.restore_latest()

# Predict
start_string = "bring"
input_eval = datasetManager.WordToNumber(start_string)
input_eval = tf.expand_dims(input_eval, 0)
text_generated = ''
hidden = [tf.zeros((1, cfg.units))]
#https://machinetalk.org/2019/03/29/neural-machine-translation-with-attention-mechanism/
predictions = model(input_eval) # why empties - we are not learning? TODO
predicted_id = tf.argmax(predictions[-1]).numpy()
text_generated += " " + datasetManager.NumberToWord([predicted_id])
print(start_string + text_generated)
