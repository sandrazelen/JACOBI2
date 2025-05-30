import time
import numpy as np
import traceback
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, OptimizeResult


def simulate_system(ode_func, X0, t, betas, method, DEBUG):
    """Simulate the system using the given parameters."""
    start_time = time.time()
    timeout = 10 # todo - need finetuning?
    
    if betas is not None and np.any(np.isnan(betas)):
        if DEBUG:
            print("NaN detected in betas:", betas)
        return None
    
    def stop_event(*args):
        ts = (time.time() - start_time)
        if ts > timeout:
            if DEBUG: print(f'solve_ivp exceeded timeout: {ts} > {timeout}')
            raise TookTooLong()
        return ts < timeout

    stop_event.terminal = True

    try:
        
        return solve_ivp(ode_func, (t[0], t[-1]), X0, t_eval=t, args=betas, method=method, events=stop_event)
    except Exception as error:
        if DEBUG: 
            print("X0 shape:", np.shape(X0))
            print("t shape:", np.shape(t))
            print("betas:", betas)
            print("simulate_system error: ", error, betas)
            print("Detailed traceback:")
            print(traceback.format_exc())
        return None


def calculate_error(simulated, observed, DEBUG):
    """Calculate the mean squared error between simulated and observed data."""
    if simulated.status == -1:
        if DEBUG: 
            """
            print("simulated y: shape ", simulated.y.T.shape)
            print(type(simulated.y.T))
            print("observed: ", observed)
            print(type(observed))
                  
            print(type(simulated.y.T))  # Should be <class 'numpy.ndarray'>
            print(simulated.y.T.dtype)  # Check data type (e.g., float64, int32)

            # Check type and dtype of observed
            print(type(observed))  # Should be <class 'tuple'>
            print(type(observed[0]))  # Check the type of the first element in the tuple
            print(len(observed[0]))
            print(observed[0].dtype)
            print("NEW: ")
            
            """
            print(simulated.y.T.shape)
            observed = np.concatenate(observed, axis=1)
            print(observed.shape)
            print(f"Shape mismatch: simulated {simulated.y.T.shape}, observed {observed.shape}. Skipping...")
        return float('inf')

    #observed = (observed - np.mean(observed, axis=0)) / np.std(observed, axis=0)
    #simulated = (simulated.y.T - np.mean(simulated.y.T, axis=0)) / np.std(simulated.y.T, axis=0)
    #print(type(simulated.y.T))
    #print(simulated.y.T.shape)
    #observed = np.concatenate(observed, axis=1)
    #print("observed beforre concat", observed.shape)
    observed = np.concatenate(observed, axis=1)
    #print(type(observed))
    #print(simulated.y.T.shape)
    #print("observed after concat", observed.shape)
    
    #observed = observed.reshape(25, -1)
    observed = observed.reshape(simulated.y.T.shape)
    if(simulated.y.T.shape != observed.shape):
        return float('inf')
    #print(observed)
    #print(simulated.y.T)
    
    obs_min, obs_max = observed.min(), observed.max()
    sim_min, sim_max = simulated.y.T.min(), simulated.y.T.max()

    observed_scaled = (observed - obs_min) / (obs_max - obs_min)
    simulated_scaled = (simulated.y.T - sim_min) / (sim_max - sim_min)
    
    #print(np.mean((simulated_scaled - observed_scaled) ** 2))
    #print(np.mean((simulated.y.T - observed) ** 2))
    #return np.mean((simulated.y.T - observed) ** 2)

    return np.mean((simulated_scaled - observed_scaled) ** 2)

def objective_function(betas, ode_func, X0, t, observed_data, ivp_method, DEBUG):
    """Objective function to minimize: the error between simulated and observed data."""
    target_betas = np.array([-0.1, -1.0, 1.1, -1.0, 0.1, 0.1, 0.005, -0.01, 0.05, 0.05])

    #w_reg=5.0
    w_reg=1.5
    #L2
    if(len(target_betas)==len(betas)):
        reg = w_reg * np.sum((betas - target_betas)**2)
    else:
        reg=100.0
    simulated = simulate_system(ode_func, X0, t, tuple(betas), ivp_method, DEBUG)
    if simulated is None:
        if DEBUG: print("SOLVE_IVP FAILED")
        return float('inf')
    #reg = w_reg * np.sum(betas ** 2)
    error = calculate_error(simulated, observed_data, DEBUG)
    if DEBUG: print(f"betas: {betas} | error: {error} | regularization: {reg}")
    return error + reg


def estimate_parameters(ode_func, X0, t, observed_data, initial_guess, min_method, ivp_method, DEBUG):
    """Estimate the parameters using scipy's minimize function."""
    best_error = float('inf')
    best_params = None

    def objective_with_tracking(params, ode_func, X0, t, observed_data, ivp_method, DEBUG):
        """Modified objective function to track the best parameters."""
        nonlocal best_error, best_params
        

        error = objective_function(params, ode_func, X0, t, observed_data, ivp_method, DEBUG)
        if error < best_error:
            best_error = error
            best_params = params.copy()
        return error

    try:
        num_params = len(initial_guess)
        bounds = [(-1.5, 1.5)] * num_params
        result = minimize(
            objective_with_tracking,
            initial_guess,
            args=(ode_func, X0, t, observed_data, ivp_method, DEBUG),
            method=min_method,
            bounds = bounds,
            # tol=1e-6,   #todo - need tuning?
            
            #options={'maxiter': 10000},  # 'disp': True, 'gtol': 1e-6, 'eps': 1e-10}
            options={'maxiter': 10000},
            callback=OptimizeStopper(DEBUG, 30)
        )
    except TookTooLong as e:
        print("Optimization terminated due to time limit")
        return OptimizeResult(fun=best_error, x=best_params, success=False, message="Optimization terminated due to time limit")

    if DEBUG: print("estimated parameters: ", result)
    return result


class TookTooLong(Warning):
    pass


class OptimizeStopper(object):
    def __init__(self, DEBUG, max_sec=60):
        self.max_sec = max_sec
        self.start = time.time()
        self.DEBUG = DEBUG

    def __call__(self, xk=None):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            if self.DEBUG: print(f"Terminating optimization: exceeded {self.max_sec} seconds")
            raise TookTooLong()