import numpy as np


def lotka_func(t, X, alpha, beta, delta, gamma):
    x, y = X
    dotx = (alpha * x) + (beta * x * y)
    doty = (delta * y) + (gamma * x * y)
    return np.array([dotx, doty])


class lotka():
    def __init__(self):
        self.func = lotka_func
        self.N = 2
        self.X0 = [1.8, 1.3]
        self.betas = [2 / 3, -4 / 3, -1, 1]


def sir_func(t, X, b1, b2, b3, b4):
    S, I, R = X
    dotS = b1 * S * I
    dotI = b2 * S * I + b3 * I
    dotR = b4 * I
    return np.array([dotS, dotI, dotR])


class SIR():
    def __init__(self):
        self.func = sir_func
        self.N = 3
        self.X0 = [997, 3, 0]
        self.betas = [-0.0002, 0.0002, -0.04, 0.04]
        
def lorenz_func(t, X, sigma, rho, beta):
    x, y, z = X
    dotx = sigma * (y - x)
    doty = x * (rho - z) - y
    dotz = x * y - beta * z
    return np.array([dotx, doty, dotz])

class lorenz():
    def __init__(self):
        self.func = lorenz_func
        self.N = 3
        self.X0 = [1.0, 1.0, 1.0]
        self.betas = [10.0, 28.0, 8/3] 

"""
def lotka_logistic_func(t, X, alpha, beta, delta, gamma, K):
    x, y = X
    dotx = alpha * x * (1 - x / K) + beta * x * y
    doty = delta * y + gamma * x * y
    return np.array([dotx, doty])

class lotka_log():
    def __init__(self):
        self.func = lotka_logistic_func
        self.N = 2  
        self.X0 = [1.8, 1.3] 
        self.betas = [2 / 3, -4 / 3, -1, 1, 50]
        
"""

def lotka_logistic_func(t, X, alpha, K, beta, delta, gamma):
    x, y = X
    dotx = alpha * x + K * x**2 + beta * x * y
    doty = delta * y + gamma * x * y
    return np.array([dotx, doty])

class lotka_log():
    def __init__(self):
        self.func = lotka_logistic_func
        self.N = 2  
        self.X0 = [1.8, 1.3] 
        self.betas = [2 / 3, -2/150, -4 / 3, -1, 1]


def hodgkin_huxley_func(t, X, I_ext, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L):
    V, m, h, n = X

    # Rate functions
    def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    def beta_m(V): return 4.0 * np.exp(-(V + 65) / 18)
    def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
    def beta_h(V): return 1 / (1 + np.exp(-(V + 35) / 10))
    def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    def beta_n(V): return 0.125 * np.exp(-(V + 65) / 80)

    # Currents
    I_Na = g_Na * (m ** 3) * h * (V - E_Na)
    I_K = g_K * (n ** 4) * (V - E_K)
    I_L = g_L * (V - E_L)

    # Differential equations
    dVdt = (I_ext - (I_Na + I_K + I_L)) / C_m
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n

    return np.array([dVdt, dmdt, dhdt, dndt])

class HodgkinHuxley():
    def __init__(self):
        self.func = hodgkin_huxley_func
        self.N = 4  # Number of equations: V, m, h, n
        self.X0 = [-65, 0.05, 0.6, 0.32]  # Initial conditions for V, m, h, n
        self.betas = [10.0, 1.0, 120.0, 36.0, 0.3, 50.0, -77.0, -54.387]
        """
        self.betas = {
            "I_ext": 10.0,  # External current
            "C_m": 1.0,  # Membrane capacitance
            "g_Na": 120.0,  # Sodium conductance
            "g_K": 36.0,  # Potassium conductance
            "g_L": 0.3,  # Leak conductance
            "E_Na": 50.0,  # Sodium reversal potential
            "E_K": -77.0,  # Potassium reversal potential
            "E_L": -54.387  # Leak reversal potential
        }    
        """

def brusselator_func(t, X, A, B):
    x, y = X
    dotx = A - (B + 1) * x + x**2 * y
    doty = B * x - x**2 * y
    return np.array([dotx, doty])

class Brusselator():
    def __init__(self):
        self.func = brusselator_func
        self.N = 2  
        self.X0 = [1.0, 1.0]  
        self.betas = [1.0, 3.0] 

def fitzhugh_nagumo_func(t, X, a, b, epsilon, I_ext):
    v, w = X
    dotv = v - (v**3 / 3) - w + I_ext
    dotw = epsilon * (v + a - b * w)
    return np.array([dotv, dotw])

class FitzHughNagumo():
    def __init__(self):
        self.func = fitzhugh_nagumo_func
        self.N = 2  # Number of equations: v, w
        self.X0 = [0.0, 0.0]  # Initial conditions for v and w
        self.betas = [0.7, 0.8, 0.08, 0.5]
        


        
    
