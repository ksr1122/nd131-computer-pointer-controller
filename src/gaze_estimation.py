from model import Model

class GazeEstimate(Model):
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extension=None, threshold=0.6):
        super(GazeEstimate, self).__init__(model_name, device, extension, threshold)

    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        self.input_name = "left_eye_image"
        self.input_shape = self.model.inputs[self.input_name].shape
        p_left_eye_image = self.preprocess_input(left_eye_image)
        p_right_eye_image = self.preprocess_input(right_eye_image)
        outputs = self.net.infer({"left_eye_image": p_left_eye_image,
                                  "right_eye_image": p_right_eye_image,
                                  "head_pose_angles": head_pose_angles})
        return self.preprocess_output(outputs[self.output_name])

    def preprocess_output(self, outputs): # output.shape: 1x3
        return(outputs[0])

