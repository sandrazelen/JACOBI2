import sympy as sp
import time

t = sp.Symbol('t')

def create_variable(i):
    return sp.Function(f'x_{i}')(t)

def diff(expr, var=t):
    return sp.diff(expr, var)

def diff2(expr, var=t):
    return sp.diff(expr, var, 2)

def o(j):
    switcher_o = {
        0: lambda a, b: a * b,  # Multiplication
        1: lambda a, b: a / b,  # Division
        2: lambda a, b: a + b,  # Addition
        3: lambda a, b: a - b   # Subtraction
    }
    return switcher_o.get(j, "Invalid")

def is_redundant(system, population):
    for p in population:
        check = True
        for (rhs1, rhs2) in zip(system, p):
            if sp.simplify(sum(rhs1[1]) - sum(rhs2[1])) != 0:
                check = False
                break
        if check:
            return True
    return False

def is_redundant_optimized(system, population):
    for p in population:
        check = True
        for rhs1, rhs2 in zip(system, p):
            if rhs1 != rhs2:
                check = False
                break
        if check:
            # print(system, p)
            return True
    return False