from model import Model

class FaceDetect(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extension=None, threshold=0.6):
        super(FaceDetect, self).__init__(model_name, device, extension, threshold)

    def predict(self, image):
        outputs = self.exec_infer(image)
        return self.preprocess_output(outputs[self.output_name])

    def preprocess_output(self, outputs):
        for box in outputs[0][0]: # output.shape: 1x1xNx7
            conf = box[2]
            if conf >= self.threshold:
                fx1 = int(box[3] * self.w)
                fy1 = int(box[4] * self.h)
                fx2 = int(box[5] * self.w)
                fy2 = int(box[6] * self.h)
                return fx1, fy1, fx2, fy2

