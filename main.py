from population.genetic_algorithm import generate_new_population
from evaluators.parameter_estimation import *
from population.initial_generation import generate_population, beautify_system
from utils.functions import get_functions, funcs_to_str
from utils.history import save_history
from utils.load_systems import create_ode_function, load_systems
from utils.mapping import get_individual_solved, add_individual_solved, \
    get_solved_map, get_term_map, convert_system_to_hash
from utils.models import SIR, lorenz, lotka, lotka_log, Brusselator, FHN, FisherKPP
from utils.plots import plot_loss_by_iteration, plot_invalid_by_iteration, \
    plot_3d_by_y, plot_lorenz_3d_estimates, plot_2d_by_func

import matplotlib.pyplot as plt
import warnings
import numpy as np 
from scipy.optimize import minimize

class Config:
    def __init__(self):
        #self.target = FisherKPP()
        self.target = FHN()
        self.G = 5 # Number of generations
        self.N = 20 # Maximum number of population
        self.M = 2 # Maximum number of equations
        self.I = 4  # Maximum number of terms per equation
        self.J = 1  # Maximum number of functions per feature
        self.allow_composite = False  # Composite Functions
        self.f0ps = get_functions("5,6,7,11, 13")
        
        self.ivp_method = 'Radau'
        #self.ivp_method = 'RK45'
        #self.ivp_method = 'BDF'
        #self.ivp_method = 'LSODA'
        self.minimize_method = 'L-BFGS-B'
        #self.minimize_method = 'COBYLA'
        #self.minimize_method = 'Powell'
       
        self.elite_rate = 0.2
        self.crossover_rate = 0.4
        self.mutation_rate = 0.6
        self.new_rate = 0.1

        self.selection_gamma = 0.9

        #self.system_load_dir = 'data/differential_equations.txt' #data/results/SIR/sir_equations.txt'
        #self.system_save_dir = 'data/differential_equations.txt'
        
        self.system_load_dir = f"C:/Users/misss/OneDrive/Desktop/JACOBI2/data/differential_equations.txt"
        self.system_save_dir = f"C:/Users/misss/OneDrive/Desktop/JACOBI2/data/differential_equations.txt"

        self.DEBUG = False
        
def main():
    #######################################################################
    #                         PARAMETERS                                  #
    ################################
    # c#######################################

    config = Config()
    
    # Time grid
    
    #t_span = np.linspace(0, 10, 15) 
    
    t_span = np.linspace(0, 1, 100) 
    spatial_grid = np.linspace(0, 5, 10)  
    
    """
    t_span = np.linspace(0, 20, 100) 
    
    spatial_grid = np.linspace(0, 30, 100)  
    """
    X0 = config.target.X0
    X0 = np.array(X0)
    print(X0)
    print("Shape of X0:", X0.shape)
    
    X0=X0.flatten()
    print("Shape of X0:", X0.shape)

    # Target data for system (Fisher-KPP or FHN)
    y_target = config.target.solve_fhn_system(t_span, spatial_grid)  
    print("Y TARGET IS : ", y_target)

    #######################################################################
    #                         TARGET DATA                                 #
    #######################################################################

    print(f"true_betas: {config.target.betas} | Initial Condition: {X0}")

    #######################################################################
    #                         INITIAL POPULATION                          #
    #######################################################################

    # Generate initial population
    population = generate_population(config)
    systems = load_systems(config.system_load_dir)

    #######################################################################
    #                         RUN                                         #
    #######################################################################

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        history = []
        time_records = []
        for i in range(config.G):
            print("#### SYSTEM EVALUATION ####")

            start_time = time.time()
            history.append([])

            for j, system in enumerate(systems):
                system_hash = convert_system_to_hash(population[j])
                solved = get_individual_solved(system_hash)

                if not solved:
                    ode_func = create_ode_function(system)

                    num_betas = sum(len(eq[1]) for eq in population[j])
                    
                    """
                    initial_guess = np.concatenate(
                        #[np.random.uniform(-1.2, 1.5, size=num_betas)]
                        #[np.random.uniform(-1.5, 1.5, size=num_betas)]
                        #[np.random.uniform(-1.0, 1.1, size=num_betas)]
                        [np.random.uniform(0.0, 0.0, size=num_betas)]
                    )
                    
                    """
                    initial_guess = np.concatenate(
                        [[-0.1, -1.0, 1.1, -1.0, 0.1, 0.1, 0.005, -0.01, 0.05, 0.05][:num_betas], np.zeros(max(0, num_betas - 10))]
                    )
                    
                    solved = estimate_parameters(ode_func, X0, t_span, y_target, initial_guess , config.minimize_method,config.ivp_method, config.DEBUG)
            
                    add_individual_solved(system_hash, solved)

                history[i].append((solved, population[j], ode_func))
                print(
                    f"### Generation {i} #{j} | Error: {round(solved.fun, 4)} | System: {beautify_system(population[j])} | Solved Parameters: {solved.x}")

            end_time = time.time()
            time_records.append(end_time - start_time)
            print(f"Completed Generation {i}: {end_time - start_time} | term_map:{len(get_term_map())}, solved_map:{len(get_solved_map())}")
            
            if i < config.G - 1:
                # Generate new population for the next generation
                population = generate_new_population(history[i], population, config)
                systems = load_systems(config.system_load_dir)

    #######################################################################
    #                         RESULT                                      #
    #######################################################################

    # Save the history and performance over time
    save_history(config, history, time_records)

    # Final best result
    best = None
    min_loss = []
    avg_loss = []
    invalid = []
    for i, generation in enumerate(history):
        loss = []
        for j, individual in enumerate(generation):
            if config.DEBUG: 
                print(
                    f'generation {i} {j} | loss:{round(individual[0].fun, 4)} func: {beautify_system(individual[1])} param:{individual[0].x}')
            loss.append(individual[0].fun)
            if best is None or individual[0].fun < best[0].fun:
                best = individual
        min_loss.append(min(loss))
        avg_loss.append(np.mean([l for l in loss if l < 100]))  # Exclude loss > 100
        invalid.append(sum(l >= 100 for l in loss))

    print(f'\nBest | Loss:{best[0].fun} func: {beautify_system(best[1])} param:{best[0].x}')

if __name__ == "__main__":
    main()