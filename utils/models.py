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

def brusselator_func(t, X, A, C, D, B, E):
    x, y = X
    dotx = A + C * x + D*x**2 * y
    doty = B * x + E* x**2 * y
    return np.array([dotx, doty])

class Brusselator():
    def __init__(self):
        self.func = brusselator_func
        self.N = 2  
        self.X0 = [1.0, 1.0]  
        self.betas = [1.0, -4.0, 1.0, 3.0, -1.0] 

def fitzhugh_nagumo_func(t, X, a, b, epsilon, I_ext):
    v, w = X
    dotv = v - (v**3 / 3) - w + I_ext
    dotw = epsilon * (v + a - b * w)
    return np.array([dotv, dotw])

class FitzHughNagumo():
    def __init__(self):
        self.func = fitzhugh_nagumo_func
        self.N = 2  
        self.X0 = [0.0, 0.0]  
        self.betas = [0.7, 0.8, 0.08, 0.5]
        
"""
class WaveEquation:
    def __init__(self, c=1.0):
        self.c = c  # Wave speed
        self.X0 = np.sin(np.pi * np.linspace(0, 1, 100))  # Initial displacement
        self.X1 = np.zeros(100)  # Initial velocity (∂u/∂t = 0)
        self.betas = [c]  # Parameter to estimate

    def func(self, t, u, v, betas):
        #∂²u/∂t² = c² ∂²u/∂x²
        dx2 = np.gradient(np.gradient(u))  # Approximation of ∂²u/∂x²
        return betas[0]**2 * dx2  # ∂²u/∂t²
"""

def fhn_system(t,U, D1, D2, a, b, epsilon):
    """
    ∂u/∂t = D1 (∂²u/∂x² + ∂²u/∂y²) + (a+1)*u^2 - a*u - u^3 + (- v)
    ∂v/∂t = D2 (∂²v/∂x² + ∂²v/∂y²) + εbu  - εv
    """
    u, v = U[0], U[1]

    d2u_dx2 = np.gradient(np.gradient(u, axis=0), axis=0)
    d2u_dy2 = np.gradient(np.gradient(u, axis=1), axis=1)
    d2v_dx2 = np.gradient(np.gradient(v, axis=0), axis=0)
    d2v_dy2 = np.gradient(np.gradient(v, axis=1), axis=1)
    
    u = np.clip(u, 1e-5, 1e2) 
    v = np.clip(v, 1e-5, 1e2) 

    #PDE 
    du_dt = D1 * (d2u_dx2 + d2u_dy2) + (a+1)*u**2 - a * u - u**3 - v
    dv_dt = D2 * (d2v_dx2 + d2v_dy2) + epsilon * (b * u - v)

    return np.array([du_dt, dv_dt])

class FHN:
    def __init__(self):
        self.func = fhn_system
        self.N = 2 
        
        #IC for u and v
        self.X0 = [
            np.random.uniform(0.1, 0.3, (100, 100)),  
            np.random.uniform(0.1, 0.3, (100, 100))  
        ]
        
        self.betas = [0.1, 0.05, 0.1, 0.5, 0.01]  # D1, D2, a, b, epsilon
        
    def solve_fhn_system(self, t_span, dt):
        u_sol = np.zeros((len(t_span), *self.X0[0].shape))  
        v_sol = np.zeros((len(t_span), *self.X0[1].shape)) 

        #IC
        u_sol[0] = self.X0[0]
        v_sol[0] = self.X0[1]

        #time stepping
        for t_idx in range(1, len(t_span)):
            t = t_span[t_idx]

            U = np.array([u_sol[t_idx-1], v_sol[t_idx-1]])
            du_dt, dv_dt = self.func(t, U, *self.betas)  # Call 

            u_sol[t_idx] = u_sol[t_idx-1] + du_dt * dt
            v_sol[t_idx] = v_sol[t_idx-1] + dv_dt * dt

        return u_sol, v_sol

"""
def fisher_kpp_func(t, U, D, r1, r2):

    u = U
    
    d2u_dx2 = np.gradient(np.gradient(u, axis=0), axis=0)
    d2u_dy2 = np.gradient(np.gradient(u, axis=1), axis=1)
    
    # Apply Fisher-KPP equation
    du_dt = D * (d2u_dx2 + d2u_dy2) + r1 * u + r2* u**2
    
    return du_dt

class FisherKPP:
    def __init__(self):
        self.func = fisher_kpp_func
        self.N = 1  # Single variable u
        
        # Initial condition: small perturbation around zero
        self.X0 = np.random.uniform(0.01, 0.02, (100, 100))
        
        self.betas = [0.1, 1.0, -1.0]  # D (diffusivity), r (growth rate), K (carrying capacity)
        
    def solve_fisher_kpp(self, t_span, dt):
        u_sol = np.zeros((len(t_span), *self.X0.shape))
        
        # Initial condition
        u_sol[0] = self.X0
        
        # Time stepping
        for t_idx in range(1, len(t_span)):
            t = t_span[t_idx]
            
            U = u_sol[t_idx-1]
            du_dt = self.func(t, U, *self.betas)  # Call equation function
            
            u_sol[t_idx] = u_sol[t_idx-1] + du_dt * dt
        
        return u_sol
"""




def fisher_kpp_system(t, U, D1, D2, r1, r2, alpha, beta):
    """
    ∂u/∂t = D1 * ∂²u/∂x² + r1 * u * (1 - u - alpha * v)
    ∂v/∂t = D2 * ∂²v/∂x² + r2 * v * (1 - v - beta * u)
    """
    u, v = U  # Extract species densities
    
    d2u_dx2 = np.gradient(np.gradient(u, axis=0), axis=0)
    d2v_dx2 = np.gradient(np.gradient(v, axis=0), axis=0)
    
    # Apply Fisher-KPP equations for u and v
    du_dt = D1 * d2u_dx2 + r1 * u * (1 - u - alpha * v)
    dv_dt = D2 * d2v_dx2 + r2 * v * (1 - v - beta * u)
    
    return np.array([du_dt, dv_dt])

"""
class FisherKPP:
    def __init__(self):
        self.func = fisher_kpp_system
        self.N = 2  # Two variables (u, v)
        
        # Smooth initial profile: Gaussian distribution or sine wave
        x = np.linspace(0, 10, 100)
        self.X0 = [
            np.exp(-0.5 * (x - 5)**2),  # u initial (Gaussian profile centered at 5)
            np.sin(x)                    # v initial (sine wave profile)
        ]
        
        # Model parameters: D1, D2, r1, r2, alpha, beta
        self.betas = [0.1, 0.2, 1.5, 1.2, 0.8, 0.6] 
    
    def solve_fisher_kpp(self, t_span, dt):
        u_sol = np.zeros((len(t_span), *self.X0[0].shape))
        v_sol = np.zeros((len(t_span), *self.X0[1].shape))
        
        # Initial conditions
        u_sol[0] = self.X0[0]
        v_sol[0] = self.X0[1]
        
        # Time stepping
        for t_idx in range(1, len(t_span)):
            t = t_span[t_idx]
            U = [u_sol[t_idx-1], v_sol[t_idx-1]]
            du_dt, dv_dt = self.func(t, U, *self.betas)  # Call system function
            
            u_sol[t_idx] = u_sol[t_idx-1] + du_dt * dt
            v_sol[t_idx] = v_sol[t_idx-1] + dv_dt * dt
        
        return u_sol, v_sol

"""

class FisherKPP:
    def __init__(self):
        self.func = fisher_kpp_system
        self.N = 2  # Two variables (u, v)
        
        # Initial conditions: small perturbation around zero
        self.X0 = [
            np.random.uniform(0.1, 0.2, (100,)),  # u initial
            np.random.uniform(0.1, 0.2, (100,))   # v initial
        ]
        
        # Model parameters: D1, D2, r1, r2, alpha, beta
        self.betas = [0.1, 0.2, 1.5, 1.2, 0.8, 0.6] 
        
    def solve_fisher_kpp(self, t_span, dt):
        u_sol = np.zeros((len(t_span), *self.X0[0].shape))
        v_sol = np.zeros((len(t_span), *self.X0[1].shape))
        
        # Initial conditions
        u_sol[0] = self.X0[0]
        v_sol[0] = self.X0[1]
        
        u_min, u_max = np.min(self.X0[0]), np.max(self.X0[0])
        v_min, v_max = np.min(self.X0[1]), np.max(self.X0[1])

        
        # Time stepping
        for t_idx in range(1, len(t_span)):
            t = t_span[t_idx]
            U = [u_sol[t_idx-1], v_sol[t_idx-1]]
            du_dt, dv_dt = self.func(t, U, *self.betas)

            u_next = u_sol[t_idx-1] + du_dt * dt
            v_next = v_sol[t_idx-1] + dv_dt * dt

            # Dynamically clip based on historical min/max
            u_next_clipped = np.clip(u_next, u_min, u_max)
            v_next_clipped = np.clip(v_next, v_min, v_max)

            u_sol[t_idx] = u_next_clipped
            v_sol[t_idx] = v_next_clipped

            # Update min/max encountered so far
            u_min = min(u_min, np.min(u_next_clipped))
            u_max = max(u_max, np.max(u_next_clipped))

            v_min = min(v_min, np.min(v_next_clipped))
            v_max = max(v_max, np.max(v_next_clipped))

            
        return u_sol, v_sol
