import os
import cv2

import logging as log

from openvino.inference_engine import IENetwork, IECore


'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class Model:
    '''
    Base Class Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, threshold):
        self.device = device
        self.extension = extension
        self.threshold = threshold
        self.model_name = model_name

        try:
            model_weights = model_name + '.bin'
            model_structure = model_name + '.xml'
            self.core = IECore()
            self.model = self.core.read_network(model_structure, model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        if extension and "CPU" in device:
            self.core.add_extension(extension, device)

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        raise NotImplementedError

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
    '''
    Before feeding the data into the model for inference,
    you might have to preprocess it. This function is where you can do that.
    '''
        raise NotImplementedError

    def preprocess_output(self, outputs):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        raise NotImplementedError
