import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel
from controllers.feedback_linearization_controller import FeedbackLinearizationController

class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3

        Model_1 = ManiuplatorModel(Tp=Tp); Model_1.m3=0.1; Model_1.r3=0.05
        Model_2 = ManiuplatorModel(Tp=Tp); Model_2.m3=0.01; Model_2.r3=0.01
        Model_3 = ManiuplatorModel(Tp=Tp); Model_3.m3=1.0; Model_3.r3=0.3
        self.Kd = FeedbackLinearizationController(Controller).Kd
        self.Kp = FeedbackLinearizationController(Controller).Kp
        self.models = [Model_1, Model_2, Model_3]
        self.i = 0
        self.Tp = Tp
        self.u = np.zeros((2, 1))

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        x_pred = np.zeros((2, 3))
        error_model = np.inf
        idx = 0

        for index, model in enumerate(self.models):
            y = model.M(x) @ self.u + model.C(x) @ np.reshape(x[2:], (2, 1))
            x_pred[0][index] = y[0]
            x_pred[1][index] = y[1]

        for model_idx in range(3):
            new_error_model = np.sum(abs(x[:2] - x_pred[:, model_idx]))
            if error_model > new_error_model:
                error_model = new_error_model
                idx = model_idx

        self.i = idx
        
        
    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        v = q_r_ddot + self.Kd @ (q_r_dot - x[2:]) + self.Kp @ (q_r - x[:2])
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        self.u = M @ v[:, np.newaxis] + C @ x[2:][:, np.newaxis]
        return self.u
