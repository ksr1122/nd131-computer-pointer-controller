import os
import cv2
import time

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
        try:
            start_time = time.time()
            self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
            log.info('Model {} Load time: {:.2f} msecs'.format(os.path.basename(self.model_name), 
                                                              (time.time()-start_time)*1000))
        except Exception as e:
            raise ValueError("Could not load the model.\n {}".format(e))

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        try:
            p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            p_frame = p_frame.transpose((2,0,1))
            p_frame = p_frame.reshape(1, *p_frame.shape)
            return p_frame
        except Exception as e:
            raise ValueError("Could not preprocess the image.\n {}".format(e))

    def preprocess_output(self, outputs):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        raise NotImplementedError
