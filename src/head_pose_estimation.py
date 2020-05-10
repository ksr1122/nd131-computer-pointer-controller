from model import Model

class HeadPoseEstimate(Model):
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extension=None, threshold=0.6):
        super(HeadPoseEstimate, self).__init__(model_name, device, extension, threshold)

    def predict(self, image):
        outputs = self.exec_infer(image)
        return self.preprocess_output(outputs)

    def preprocess_output(self, outputs): # output.shape: "name": 1x1
        angle_y_fc = outputs["angle_y_fc"][0][0]
        angle_p_fc = outputs["angle_p_fc"][0][0]
        angle_r_fc = outputs["angle_r_fc"][0][0]
        return angle_y_fc, angle_p_fc, angle_r_fc

