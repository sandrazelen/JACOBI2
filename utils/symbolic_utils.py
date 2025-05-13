import sympy as sp
import time

t = sp.Symbol('t')
x = sp.Symbol('x')
y = sp.Symbol('y')
x_1 = sp.Symbol('x_1')
x_2 = sp.Symbol('x_2')

"""
def create_variable(i):
    return sp.Function(f'x_{i}')(t, x)
"""

def create_variable(i):
    return sp.Function(f'x_{i}')(t, x, y)


"""
def create_variable(i):
    return sp.Function(f'x_{i}')(t, x_1, x_2)
"""


def diff(expr, var=t):
    return sp.diff(expr, var)

def diff2(expr, var=x):
    return sp.diff(expr, var, 2)

def diffy(expr, var=y):
    return sp.diff(expr, var, 2)

def diff_x1(expr):
    return sp.diff(expr, x_1)

def diff2_x1(expr):
    return sp.diff(expr, x_1, 2)

def diff_x2(expr):
    return sp.diff(expr, x_2)

def diff2_x2(expr):
    return sp.diff(expr, x_2, 2)

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
