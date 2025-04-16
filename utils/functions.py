import sympy as sp


def one(x):
    return 1


def sin(x):
    return sp.sin(x)


def cos(x):
    return sp.cos(x)


def tan(x):
    return sp.tan(x)


def exp(x):
    return sp.exp(x)


def linear(x):
    return x


def square(x):
    return x ** 2


def cube(x):
    return x ** 3


def quart(x):
    return x ** 4


def log(x):
    return sp.log(x)


"""
all_functions = {
    0: ("one", one),
    1: ("sin", sin),
    2: ("cos", cos),
    3: ("tan", tan),
    4: ("exp", exp),
    5: ("linear", linear),
    6: ("square", square),
    7: ("cube", cube),
    8: ("quart", quart),
    9: ("log", log)
}
"""

x, y, t = sp.symbols('x y t')

def diff_x(expr):
    return sp.diff(expr, x)

def diff2_x(expr):
    return sp.diff(expr, x, 2)

def diff_y(expr):
    return sp.diff(expr, y)


def diff2_y(expr):
    return sp.diff(expr, y, 2)

def diff_t(expr):
    return sp.diff(expr, t)

def diff2_t(expr):
    return sp.diff(expr, t, 2)



all_functions = {
    0: ("one", one),
    1: ("sin", sin),
    2: ("cos", cos),
    3: ("tan", tan),
    4: ("exp", exp),
    5: ("linear", linear),
    6: ("square", square),
    7: ("cube", cube),
    8: ("quart", quart),
    9: ("log", log),
    10: ("diff_x", diff_x),       # First derivative w.r.t x
    11: ("diff2_x", diff2_x),     # Second derivative w.r.t x
    12: ("diff_y", diff_y),       # First derivative w.r.t y
    13: ("diff2_y", diff2_y)
}


def get_allowed_functions():
    print("Available functions:")
    for key, (name, _) in all_functions.items():
        print(f"{key}: {name}")

    # allowed_functions = input("Enter the numbers of the functions you want to allow (comma-separated): ")
    allowed_functions = "5"
    return [int(x.strip()) for x in allowed_functions.split(',')]


def get_functions(selected_functions):
    selected_functions = selected_functions.replace("[", "").replace("]", "")
    funcs = [int(x.strip()) for x in selected_functions.split(',')]
    return funcs


def funcs_to_str(funcs):
    func_str = []
    for func in funcs:
        func_str.append(all_functions[func][0])
    func_str = ','.join(func_str)
    return func_str


def f(i):
    return all_functions.get(i, ("Invalid", lambda x: x))[1]
