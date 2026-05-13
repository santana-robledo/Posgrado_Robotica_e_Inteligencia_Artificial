# npid_indirecto.py
import numpy as np

class nPID_Indirecto:
    def __init__(self, n=4, beta=1.0):
        self.n = n
        self.beta = beta
        self.w = 0.1 * np.random.randn(n, 1)  # Iniciar pequeños
        self.y = 0.0
        self.P = np.eye(n) * 10.0
        self.Q = np.eye(n) * 0.01
        self.R = 0.1
        self.alpha = 0.05

    def control_u(self, x, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        x = x.reshape((self.n, 1))
        v = float(np.dot(self.w.T, x))
        self.y = np.tanh(v * self.alpha)
        return self.y * self.beta

    def fit(self, error_trayectoria, x, eta=0.01):
        x = x.reshape((self.n, 1))
        H = self.get_H(x).reshape((self.n, 1))
        PH = np.dot(self.P, H)
        denom = self.R + np.dot(H.T, PH)
        k = PH / denom
        self.w += eta * k * error_trayectoria
        self.P = self.P - np.dot(k, H.T.dot(self.P)) + self.Q

    def get_H(self, x):
        tanh_deriv = (1 - self.y ** 2)
        return tanh_deriv * x.flatten() * self.beta * self.alpha