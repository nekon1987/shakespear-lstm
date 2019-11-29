import tensorflow as tf
from modules.configuration import Configuration
from modules.diagnostics import Logger

class InfrastructureService(object):

    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def SetupTensorflowForGpuMode(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000, )]) #todo to config
        Logger.Log('GPU-Mode enabled')