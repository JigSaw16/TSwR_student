import numpy as np

from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        p_1, p_2 = p[0], p[1]
        self.L = np.array([[3*p_1, 0], [0, 3*p_2], [3*p_1**2, 0], [0, 3*p_2**2], [p_1**3, 0], [0, p_2**3]])
        W = np.array([[1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.]])
        A = np.array([[0., 0., 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 1., 0.],
                        [0., 0., 0., 0., 0., 1.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]])
        B = np.zeros((6, 2))
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        A = np.array([[0., 0., 1., 0., 0., 0.],
              [0., 0., 0., 1., 0., 0.],
              [0., 0., 0., 0., 1., 0.],
              [0., 0., 0., 0., 0., 1.],
              [0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0.]])
        
        x = np.concatenate([q, q_dot], axis=0)
        Minv = np.linalg.inv(self.model.M(x))
        C = self.model.C(x)
        MinvC = -(Minv @ C)
        A[2:4, 2:4] = MinvC[0:2, 0:2]

        B = np.zeros((6, 2))
        B[2:4, 0:2] = MinvC[0:2, 0:2]

        self.eso.A = A
        self.eso.B = B


    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        q = x[:2]
        M = self.model.M(x)
        C = self.model.C(x)
        z_approx = self.eso.get_state()
        x_approx, x_approx_dot, f = z_approx[:2], z_approx[2:4], z_approx[4:]
        e = q_d - q
        e_dot = q_d_dot - x_approx_dot
        v = q_d_ddot + self.Kd @ e_dot + self.Kp @ e
        u = M @ (v - f) + C @ x_approx_dot
        self.update_params(x_approx, x_approx_dot)
        self.eso.update(q.reshape(len(q), 1), u.reshape(len(u), 1))
        return u
        # return NotImplementedError
