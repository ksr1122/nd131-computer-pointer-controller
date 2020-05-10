from model import Model

class FacialLandMarkDetect(Model):
    '''
    Class for the Facial Landmark Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extension=None, threshold=0.6):
        super(FacialLandMarkDetect, self).__init__(model_name, device, extension, threshold)

    def predict(self, image):
        outputs = self.exec_infer(image)
        return self.preprocess_output(outputs[self.output_name])

    def preprocess_output(self, outputs): # output.shape: 1x10x1x1
        ex1, ey1 = int(outputs[0][0][0][0] * self.w), int(outputs[0][1][0][0] * self.h)
        ex2, ey2 = int(outputs[0][2][0][0] * self.w), int(outputs[0][3][0][0] * self.h)
        return ex1, ey1, ex2, ey2

