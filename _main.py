import tensorflow as tf

from modules.tensorboard import TensorboardManager
from shakespeare_model import ShakespeareModel
from modules.infrastructure import InfrastructureService
from modules.configuration import ConfigurationProvider
from modules.datasets import DatasetManager
from training_supervisor import TrainingSupervisor

cfg = ConfigurationProvider.CreateConfiguration()
tensorboardManager = TensorboardManager(cfg) #todo naming consistency
infrastructureService = InfrastructureService(cfg)
datasetManager = DatasetManager(cfg)
trainingSupervisor = TrainingSupervisor(cfg, tensorboardManager)

infrastructureService.SetupTensorflowForGpuMode()
tensorboardManager.SetupTensorboard()
vocabulary_size, training_dataset = datasetManager.CreateTrainingSet()

model = ShakespeareModel(vocabulary_size, cfg.embedding_dim, cfg.units, cfg.batch_size)

trainingSupervisor.configure_checkpoints(model)
trainingSupervisor.train_model(model, training_dataset)
trainingSupervisor.restore_latest()


# Predict

def test_model(start_string, length_of_sequence_to_Generate):
    def generate_single_word(phrase):
        input_eval = datasetManager.WordToNumber(phrase)
        input_eval = tf.expand_dims(input_eval, 0)
        #https://machinetalk.org/2019/03/29/neural-machine-translation-with-attention-mechanism/
        predictions, previous_state = model(input_eval, None) # why empties - we are not learning? TODO
        predicted_id = tf.argmax(predictions[-1]).numpy()
        return datasetManager.NumberToWord(predicted_id)

    last_generated_word = start_string
    sentence = start_string
    for i in range(length_of_sequence_to_Generate):
        last_generated_word = generate_single_word(last_generated_word)
        sentence += ' ' + last_generated_word

    print(sentence)

test_model('back', 8)
test_model('against', 8)
test_model('but', 8)
test_model('he', 8)
test_model('that', 8)
test_model('for', 8)
test_model('was', 8)
test_model('we', 8)
test_model('to', 8)
test_model('partly', 8)
test_model('cannot', 8)
test_model('always', 8)
test_model('they', 8)
test_model('most', 8)
test_model('not', 8)
test_model('natural', 8)
test_model('all', 8)
test_model('how', 8)
test_model('their', 8)
test_model('great', 8)
test_model('stiff', 8)
test_model('must', 8)
test_model('make', 8)



test_model('make')
