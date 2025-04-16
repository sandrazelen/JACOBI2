import copy
import random as rd
import sympy as sp
from utils.symbolic_utils import t, x_1,x_2, x, create_variable, o, diff2
from utils.functions import f
from utils.numpy_conversion import save_systems_as_numpy_funcs
from utils.mapping import convert_system_to_hash

#remember to import x_2 when needed

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


"""
def generate_systems(N, config):
    variables = [create_variable(i) for i in range(1, config.M + 1)]
    v = copy.deepcopy(variables)
    print(variables)

    systems_hash_list = []
    systems = []
    n = 0
    
    while n < N:
        system = []
        valid_system = True
        
        for m in range(config.M):
            terms = []
            if rd.random() < 0.5:
                terms.append(diff2_x1(variables[m]))
            if rd.random()<0.5:
                terms.append(diff2_x2(variables[m]))
                
            #for i in range(config.I-2):
            while len(terms)<4:
                term = generate_term(v, config, True, equation_idx=m)
                if term is not None and not term in terms:
                    terms.append(term)
            
            
            non_zero_terms = [t for t in terms if t != 0 and t!=None]
            non_zero_terms.sort(key=lambda x: len(str(x)))
            
            print(non_zero_terms)
            system.append([sp.diff(variables[m], t), non_zero_terms])
            #print(system)
                
            s_hash = convert_system_to_hash(system)
            if s_hash not in systems_hash_list:        
                systems.append(system)
                systems_hash_list.append(s_hash)
                n += 1
        
    return systems
"""

def generate_systems(N, config):
    variables = [create_variable(i) for i in range(1, config.M + 1)]
    v = copy.deepcopy(variables)
    print(variables)
    print(v)
    systems_hash_list = []
    systems = []
    n = 0
    total_attempts = 0
    while n < N and total_attempts < 100:
        total_attempts += 1
        system = []
        valid_system = True
        required_terms = [4,4]  
        
        for m in range(config.M):
            terms = []
            
            if rd.random() < 0.99 and m==0:
                #terms.append(diff2_x1(variables[m]))
                terms.append(diff2(variables[m]))
            if rd.random()<0.99 and m==1:
                terms.append(diff2(variables[m]))
                
            if rd.random()<0.99:
                operator = o(0)
                terms.append(operator(variables[0], variables[1]))
                
            #print("TERMS: ", terms)
            
            attempts = 0
            while len([t for t in terms if t != 0]) < required_terms[m] and attempts < 1000:
                attempts += 1
                term = generate_term(v, config, True, equation_idx=m)  # Pass equation index
                if term is not None and term != 0 and term not in terms:
                    terms.append(term)
            
            non_zero_terms = [t for t in terms if t != 0]
            non_zero_terms.sort(key=lambda x: len(str(x)))
            if len(non_zero_terms) == required_terms[m]:
                system.append([sp.diff(variables[m], t), non_zero_terms])
            else:
                valid_system = False
                break
        
        if valid_system and len(system) == config.M:
            s_hash = convert_system_to_hash(system)
            if s_hash not in systems_hash_list:
                systems.append(system)
                systems_hash_list.append(s_hash)
                n += 1
        
        if total_attempts >= 1000:
            break

    return systems


"""
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

"""

def generate_term(variables, config, non_empty, equation_idx=0):
    term = None
    var_list = []
    j = 0
    
    """
    # Define allowed functions for each equation
    if equation_idx == 0:  # First equation: allow linear, quadratic
        allowed_ops = [op for op in config.f0ps if op in [5, 6]]  # 5=linear, 6=quadratic, 7=cubic, 11,13=derivatives
        weights = [0.5 if op==5 else 0.2 if op in [5,6] else 0.2 for op in allowed_ops]
    else:  # Second equation: only allow linear terms
        allowed_ops = [op for op in config.f0ps if op in [5, 6]]  # 5=linear, 11,13=derivatives
        weights = [0.5 if op==5 else 0.4 for op in allowed_ops]
    """
    
    allowed_ops = [op for op in config.f0ps if op in [5, 6]]  # 5=linear, 11,13=derivatives
    weights = [0.5 for op in allowed_ops]
    
    if non_empty or rd.randint(0, 99) < 90:
        chosen_vars = []
        for var in variables:
            #print("VAR IS: ", var)
            base_name = str(var.func).split('(')[0]
            if rd.random() < 0.5:
                chosen_vars.append(create_variable(int(base_name.split('_')[1])))
            else:
                chosen_vars.append(create_variable(2))
            
        for var in chosen_vars:
            if j == 0 or rd.randint(0, 99) < 90:
                func = f(rd.choices(allowed_ops, weights=weights, k=1)[0])
                var = func(var)
                j += 1
                var_list.append(var)
                if j == config.J:
                    break

    if var_list:
        term = var_list[0]

    
    return term


"""


def generate_term(variables, config, non_empty):
    rd.shuffle(variables)
    term = None
    var_list = []
    j = 0

    #weights = [1 / len(config.f0ps)] * len(config.f0ps) # todo - term distribution
    weights = [0.5 if op == 5 else 0.5 / (len(config.f0ps) - 1) for op in config.f0ps]
    if non_empty or rd.randint(0, 99) < 75: # equation to have at least one term
        for var in variables:
            if j == 0 or rd.randint(0, 99) < 25:
                func = f(rd.choices(config.f0ps, weights=weights, k=1)[0])# todo
                var = func(var)
                j += 1

                var_list.append(var)
                if j == config.J: break

    if var_list:
        term = var_list[0]
        for var in var_list[1:]:
            operator = o(0)
            term = operator(term, var)
    print(term)
    return term



def generate_term(variables, config, non_empty, equation_idx=0):
    term = None
    var_list = []
    j = 0
    
    # Define allowed functions for each equation
    if equation_idx == 0:  # First equation: allow linear, quadratic, cubic
        #allowed_ops = [op for op in config.f0ps if op in [5, 6, 7, 11, 13]]  # 5=linear, 6=quadratic, 7=cubic, 11,13=derivatives
        weights = [0.3 if op==5 else 0.2 if op in [6,7] else 0.1 for op in config.f0ps]
    else:  # Second equation: only allow linear terms
        #allowed_ops = [op for op in config.f0ps if op in [5, 11, 13]]  # 5=linear, 11,13=derivatives
        weights = [0.5 if op==5 else 0.1 for op in config.f0ps]
    
    if non_empty or rd.randint(0, 99) < 90:
        chosen_vars = []
        for var in variables:
            base_name = str(var.func).split('(')[0]
            if rd.random() < 0.9:
                chosen_vars.append(create_variable(int(base_name.split('_')[1])))
            else:
                chosen_vars.append(create_variable(2))
            
        for var in chosen_vars:
            if j == 0 or rd.randint(0, 99) < 25:
                func = f(rd.choices(config.f0ps, weights=weights, k=1)[0])
                var = func(var)
                j += 1
                var_list.append(var)
                if j == config.J:
                    break

    if var_list:
        term = var_list[0]
    
    return term

"""
def beautify_equation(eq, beta_start):
    lhs = eq[0]
    # Extract just the function name (x_i) without arguments for the LHS
    func_name = str(lhs.args[0].func).split('(')[0]
    lhs_str = f"d{func_name}/dt"
    
    replacements = {
        "sin": "sin", "cos": "cos", "tan": "tan", "exp": "exp",
        "**2": "²", "**3": "³", "**4": "⁴",
        "Derivative": "d",
        "(t, x_1, x_2)": "",  # Remove function arguments
        "(t)": "",
        "d(x_": "d²x_",  # Handle second derivatives more elegantly
        ", (x_1, 2))": "/dx_1²",
        ", (x_2, 2))": "/dx_2²",
        ", (t, 2))": "/dt²"
    }

    for i in range(1, 100):
        replacements[f"x_{i}(t, x_1, x_2)"] = f"x_{i}"
        replacements[f"x_{i}(t, x)"] = f"x_{i}"
        replacements[f"x_{i}(t)"] = f"x_{i}"
        replacements[f"Derivative(x_{i}(t, x_1, x_2), t)"] = f"dx_{i}/dt"
        replacements[f"Derivative(x_{i}(t, x_1, x_2), x_1)"] = f"dx_{i}/dx_1"
        replacements[f"Derivative(x_{i}(t, x_1, x_2), x_2)"] = f"dx_{i}/dx_2"
        replacements[f"Derivative(x_{i}(t), (t, 2))"] = f"d²x_{i}/dt²"
        replacements[f"Derivative(x_{i}(t, x_1, x_2), (x_1, 2))"] = f"d²x_{i}/dx_1²"
        replacements[f"Derivative(x_{i}(t, x_1, x_2), (x_2, 2))"] = f"d²x_{i}/dx_2²"
        replacements[f"Derivative(x_{i}(t, x), (x, 2))"] = f"dx_{i}/dx²"

    beautified_terms = []
    beta_count = beta_start
    if eq[1]:
        # First beautify all terms
        for term in eq[1]:
            if term == 0:  # Skip zero terms completely
                continue
            term_str = str(term)
            for old, new in replacements.items():
                term_str = term_str.replace(old, new)
            beautified_terms.append((term_str, beta_count))
            beta_count += 1
        
        # Sort by length of the term (not including the beta part)
        beautified_terms.sort(key=lambda x: len(x[0]))
        # Convert to final format after sorting
        beautified_terms = [f"beta_{beta}*({term})" for term, beta in beautified_terms]

    rhs_str = " + ".join(beautified_terms)
    if not beautified_terms:  # Only if there are no terms at all
        rhs_str = "0"
    
    return f"{lhs_str} = {rhs_str}"

def beautify_system(system):
    beta_start = 0
    equations = []
    for eq in system:
        equations.append(beautify_equation(eq, beta_start))
        beta_start += len(eq[1])
    return equations

