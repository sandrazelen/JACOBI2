import copy
import random as rd
import sympy as sp
from utils.symbolic_utils import t, create_variable, o
from utils.functions import f
from utils.numpy_conversion import save_systems_as_numpy_funcs

from utils.mapping import convert_system_to_hash


def generate_population(config):
    print("\n#### GENERATE INITIAL POPULATION ####")
    systems = generate_systems(config.N, config)
    for i, system in enumerate(systems):
        if config.DEBUG: print(f"generate_system {i}: {system}")

    beautified_systems = [beautify_system(p) for p in systems]
    for i, system in enumerate(beautified_systems):
        print(f"beautified_systems {i}: {system}")

    # Save the system as NumPy functions
    save_systems_as_numpy_funcs(systems, config.system_save_dir)
    print(f"System saved as NumPy functions in {config.system_save_dir}\n")

    return systems


def generate_systems(N, config):
    variables = [create_variable(i) for i in range(1, config.M + 1)]
    v = copy.deepcopy(variables)

    systems_hash_list = []
    systems = []
    n = 0
    while n < N:
        system = []
        for m in range(config.M):
            terms = []
            for i in range(config.I):
                term = generate_term(v, config, i == 0)
                if term is not None and not term in terms:
                    terms.append(term)
            system.append([sp.diff(variables[m], t), terms])

        s_hash = convert_system_to_hash(system)
        if not s_hash in systems_hash_list:
            systems.append(system)
            systems_hash_list.append(s_hash)
            n += 1

    return systems


def generate_term(variables, config, non_empty):
    rd.shuffle(variables)
    term = None
    var_list = []
    j = 0

    #SANDRA: ADDED THIS
    sigma_target = 10
    rho_target = 28
    beta_target = 8 / 3

    # Define weights inversely proportional to the parameters, normalized to sum to 1
    total = sigma_target + rho_target + beta_target
    weights = [sigma_target / total, rho_target / total, beta_target / total]
    num_f0ps = len(config.f0ps)  # Get the length of config.f0ps
    if num_f0ps > 3:
        # If there are more than 3 elements, distribute weights proportionally across the first 3, and assign 0s to the rest
        weights += [0] * (num_f0ps - 3)
    elif num_f0ps < 3:
        # If there are fewer than 3 elements, we can just scale the Lorenz weights to match the number of elements
        scale_factor = 3 / num_f0ps
        weights = [w * scale_factor for w in weights[:num_f0ps]]

    
    #SANDRA: COMMENTED THIS OUT  
    #weights = [1 / len(config.f0ps)] * len(config.f0ps) # todo - term distribution /  [0.5 if op == 5 else 0.2 / (len(config.f0ps) - 1) for op in config.f0ps]
    
    if non_empty or rd.randint(0, 1) == 1: # equation to have at least one term
        for var in variables:
            if j == 0 or rd.randint(0, 1) == 1:

                func = f(rd.choices(config.f0ps, weights=weights, k=1)[0])# todo
                var = func(var)
                j += 1
                if config.allow_composite and j < config.J:  # limit applying composite to only once
                    if rd.randint(0, 1) == 1:
                        func = f(rd.choices(config.fOps, weights=weights, k=1)[0])
                        var = func(var)
                        j += 1
                var_list.append(var)
                if j == config.J: break

    if var_list:
        term = var_list[0]
        for var in var_list[1:]:
            operator = o(rd.randint(0, 1))
            term = operator(term, var)

    return term


def beautify_equation(eq, beta_start):
    lhs = eq[0]
    lhs_str = f"d{str(lhs.args[0])}/dt"

    replacements = {
        "sin": "sin", "cos": "cos", "tan": "tan", "exp": "exp",
        "**2": "²", "**3": "³", "**4": "⁴",
        "Derivative": "d",
        "(t)": "",
    }

    for i in range(1, 100):  # Adjust range as needed
        replacements[f"x_{i}(t)"] = f"x_{i}"
        replacements[f"Derivative(x_{i}(t), t)"] = f"dx_{i}/dt"
        replacements[f"Derivative(x_{i}(t), (t, 2))"] = f"d²x_{i}/dt²"

    # Split the right-hand side into individual terms
    beautified_terms = []
    if not eq[1]:
        beautified_terms.append('0')
    else:
        for i, term in enumerate(eq[1]):
            term_str = str(term)
            for old, new in replacements.items():
                term_str = term_str.replace(old, new)
            beautified_terms.append(f"beta_{beta_start + i}*({term_str})")

    rhs_str = " + ".join(beautified_terms)

    return f"{lhs_str} = {rhs_str}"


def beautify_system(system):
    beta_count = 0
    beautified_equations = []
    for eq in system:
        beautified_eq = beautify_equation(eq, beta_count)
        beautified_equations.append(beautified_eq)
        beta_count += len(eq[1])

    return beautified_equations


def manual_sir_systems():
    variables = [create_variable(i) for i in range(1, 4)]
    linear = f(5)
    mult = o(0)
    div = o(1)

    system0 = [
        [sp.diff(variables[0], t),
         [
             mult(linear(variables[0]), linear(variables[1]))
         ]
         ],
        [sp.diff(variables[1], t),
         [
             mult(linear(variables[0]), linear(variables[1])),
             linear(variables[1]),
         ]
         ],
        [sp.diff(variables[2], t),
         [
             linear(variables[1])
         ]
         ],
    ]
    return [system0]

def manual_lotka_systems():
    variables = [create_variable(i) for i in range(1, 3)]
    linear = f(5)
    mult = o(0)
    div = o(1)

    system0 = [
        [sp.diff(variables[0], t),
         [
             linear(variables[0]),
             mult(linear(variables[0]), linear(variables[1]))
         ]
         ],
        [sp.diff(variables[1], t),
         [
             linear(variables[1]),
             mult(linear(variables[0]), linear(variables[1]))
         ]
         ],
    ]

    system1 = [
        [sp.diff(variables[0], t),
         [
             linear(variables[0]),
             linear(variables[1])
         ]
         ],
        [sp.diff(variables[1], t),
         [
             linear(variables[0]),
             mult(linear(variables[0]), linear(variables[1]))
         ]
         ]
    ]

    system2 = [
        [sp.diff(variables[0], t),
         [
             div(linear(variables[0]), linear(variables[1]))
         ]
         ],
        [sp.diff(variables[1], t),
         [
             linear(variables[0]),
             linear(variables[1])
         ]
         ]
    ]

    system3 = [
        [sp.diff(variables[0], t),
         [
             mult(linear(variables[0]), linear(variables[1]))
         ]
         ],
        [sp.diff(variables[1], t),
         [
             div(linear(variables[0]), linear(variables[1])),
         ]
         ]
    ]

    system4 = [
        [sp.diff(variables[0], t),
         [
             linear(variables[0]),
             div(linear(variables[0]), linear(variables[1]))
         ]
         ],
        [sp.diff(variables[1], t),
         [
             linear(variables[0]),
             mult(linear(variables[0]), linear(variables[1]))
         ]
         ]
    ]
    return [system0, system1, system2, system3]

    # return [system0, system1, system2, system3, system4]
