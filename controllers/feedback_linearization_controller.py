import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)
        self.Kd = np.array([[50, 0], [0, 25]])
        self.Kp = np.array([[15, 0], [0, 70]])

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q = x[:2].reshape(2,1)
        q_dot = x[2:].reshape(2,1)
        q_r_ddot = q_r_ddot.reshape(2,1)

        v = q_r_ddot + self.Kd @ (q_r_dot.reshape(2,1) - q_dot) + self.Kp @ (q_r.reshape(2,1) - q)
        tau = self.model.M(x) @ v + self.model.C(x) @ q_r_dot.reshape(2,1)
        
        return tau
