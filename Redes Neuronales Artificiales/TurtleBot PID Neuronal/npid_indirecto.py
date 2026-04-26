import numpy as np

class nPID_Indirecto:
    def __init__(self, n=3, beta=1):
        self.w_c = np.random.rand(n, 1) # Red Controladora
        self.w_i = np.random.rand(n, 1) # Red Identificadora
        
        self.beta = beta
        self.y_control = 0
        self.y_identificador = 0
        
        self.P_c = np.diag(np.ones(n))*10
        self.Q_c = np.diag(np.ones(n))
        self.R_c = 0.1
        
        self.P_i = np.diag(np.ones(n))*10
        self.Q_i = np.diag(np.ones(n))
        self.R_i = 0.1

    def control_u(self, x, alpha=0.05):
        self.alpha = alpha
        v_c = np.dot(self.w_c.T, x)
        self.y_control = np.tanh(v_c * self.alpha)
        return self.y_control * self.beta
        
    def fit(self, error_trayectoria, x, u_aplicado, salida_real, eta_c=0.01, eta_i=0.01):
        
        # 1. IDENTIFICADOR
        v_i = np.dot(self.w_i.T, x)
        self.y_identificador = np.tanh(v_i * self.alpha)
        error_identificacion = salida_real - self.y_identificador
        
        H_i = ((1 - self.y_identificador**2) * x * self.alpha).reshape((3,1))
        PH_i = np.dot(self.P_i, H_i)
        inv_i = np.linalg.inv(self.R_i + np.dot(H_i.T, PH_i))
        k_i = np.dot(PH_i, inv_i)
        
        self.w_i = self.w_i + eta_i * np.dot(k_i, error_identificacion)
        self.P_i = self.P_i - np.dot(k_i, np.dot(H_i.T, self.P_i)) + self.Q_i
        
        # 2. CONTROLADOR
        dy_du = (1 - self.y_identificador**2) * self.w_i.mean() * self.alpha
        error_retropropagado = error_trayectoria * dy_du
        
        H_c = ((1 - self.y_control**2) * x * self.beta * self.alpha).reshape((3,1))
        PH_c = np.dot(self.P_c, H_c)
        inv_c = np.linalg.inv(self.R_c + np.dot(H_c.T, PH_c))
        k_c = np.dot(PH_c, inv_c)
        
        self.w_c = self.w_c + eta_c * np.dot(k_c, error_retropropagado)
        self.P_c = self.P_c - np.dot(k_c, np.dot(H_c.T, self.P_c)) + self.Q_c
