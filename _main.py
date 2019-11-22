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

trainingSupervisor.configure_checkpoints(model)
trainingSupervisor.train_model(model, training_dataset)
trainingSupervisor.restore_latest()

# Predict

def test_model(start_string):
    input_eval = datasetManager.WordToNumber(start_string)
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = ''
    hidden = [tf.zeros((1, cfg.units))]
    #https://machinetalk.org/2019/03/29/neural-machine-translation-with-attention-mechanism/
    predictions, previous_state = model(input_eval, None) # why empties - we are not learning? TODO
    predicted_id = tf.argmax(predictions[-1]).numpy()
    text_generated += " " + datasetManager.NumberToWord(predicted_id)
    print(start_string + text_generated)

test_model('back')
test_model('against')
test_model('but')
test_model('he')
test_model('that')
test_model('for')
test_model('was')
test_model('we')
test_model('to')
test_model('partly')
test_model('cannot')
test_model('always')
test_model('they')
test_model('most')
test_model('not')
test_model('natural')
test_model('all')
test_model('how')
test_model('their')
test_model('great')
test_model('stiff')
test_model('must')
test_model('make')

test_model('make')
