import sympy as sp
import numpy as np

def eq_to_numpy_func(eq, betas):
    rhs = eq.rhs
    variables = list(rhs.free_symbols)
    variables.sort(key=lambda x: x.name)
    lambda_func = sp.lambdify(variables + betas, rhs, modules=['numpy'])

    def numpy_func(X, betas):
        return lambda_func(*X, *betas)

    return numpy_func


def save_str_systems_as_numpy_funcs(systems, filename):
    with open(filename, 'w') as f:
        f.write("import numpy as np\n\n")
        for n, system in enumerate(systems):
            system = system['System']
            M = len(system)
            j = 0
            for i, eq in enumerate(system):
                lhs = eq.split('=')[0]
                var = str(eq.split('=')[1])

                f.write(f"def eq_{n}_{i}(X, betas, t):\n")
                f.write(f"    # d{var}/dt = {eq[2]}\n")
                f.write("    ")
                f.write(", ".join([f"x_{j + 1}" for j in range(M)]))  # Using j+1 to start from x_1
                f.write(" = [X[0:100], X[100:]]\n")

                terms = []
                if not eq[1]:
                    terms.append('0')
                else:
                    for term in eq[1]:
                        term_str = str(term)
                        term_str = term_str.replace('x_1(t, x)', 'x_1').replace('x_2(t, x)', 'x_2').replace('x_3(t, x)',
                                                                                                      'x_3').replace(
                            'x_4(t, x)', 'x_4').replace('x_5(t, x)', 'x_5')
                                                                                                      
                        term_str = term_str.replace('x_1(t, x, y)', 'x_1').replace('x_2(t, x, y)', 'x_2')
                        term_str = term_str.replace('sin', 'np.sin').replace('cos', 'np.cos').replace('tan',
                                                                                                      'np.tan').replace(
                            'exp', 'np.exp').replace(
                            'log', 'np.log')
                        terms.append(f"betas[{j}] * ({term_str})")
                        j += 1

                rhs_str = " + ".join(terms)
                f.write(f"    return {rhs_str}\n\n")

            f.write(f"def system_{n}(X, betas, t):\n")
            f.write(
                "    return np.array([" + ", ".join(f"eq_{n}_{i}(X, betas, t)" for i in range(len(system))) + "])\n\n")


def save_systems_as_numpy_funcs(systems, filename):
    with open(filename, 'w') as f:
        f.write("import numpy as np\n\n")

        for n, system in enumerate(systems):
            M = len(system)
            j = 0
            for i, eq in enumerate(system):
                lhs = eq[0]
                var = str(lhs.args[0].func)


                f.write(f"def eq_{n}_{i}(X, betas, t):\n")
                f.write(f"    # d{var}/dt = {eq[1]}\n")
                
                f.write("    X = X.reshape({}, 10, 10)\n".format(M))
                #JUST ADDED THE FOLLOWING LINE BELOW
                #f.write("    X = np.clip(X, -1e3, 1e3)\n") 
                f.write("    ")
                f.write(", ".join([f"x_{j + 1}" for j in range(M)]))  # Using j+1 to start from x_1
                
                
                #f.write(" = [X[0:100], X[100:]]\n")
                f.write(" = X[0], X[1]\n")

                terms = []
                if not eq[1]:
                    terms.append('0')
                else:
                    for term in eq[1]:
                        term_str = str(term)
                        # Replace function calls with array references
                        term_str = term_str.replace('x_1(t, x_1, x_2)', 'x_1')
                        term_str = term_str.replace('x_2(t, x_1, x_2)', 'x_2')
                        term_str = term_str.replace('x_1(t, x)', 'x_1')
                        term_str = term_str.replace('x_2(t, x)', 'x_2')
                        term_str = term_str.replace('x_1(t, x, y)', 'x_1')
                        term_str = term_str.replace('x_2(t, x, y)', 'x_2')

                        # Handling derivatives and converting them to numpy expressions
                        term_str = term_str.replace('Derivative(x_1(t, x_1, x_2), (x_1, 2))', "derivatives['du_dxx']")
                        term_str = term_str.replace('Derivative(x_1(t, x_1, x_2), (x_2, 2))', "derivatives['du_dyy']")
                        term_str = term_str.replace('Derivative(x_2(t, x_1, x_2), (x_1, 2))', "derivatives['dv_dxx']")
                        term_str = term_str.replace('Derivative(x_2(t, x_1, x_2), (x_2, 2))', "derivatives['dv_dyy']")
                        
                        term_str = term_str.replace('Derivative(x_1(t, x), (x, 2))', "np.gradient(np.gradient(x_1, axis=0))")
                        term_str = term_str.replace('Derivative(x_2(t, x), (x, 2))', "np.gradient(np.gradient(x_2, axis=0))")
                        term_str = term_str.replace('Derivative(x_1, (x, 2))', "np.gradient(np.gradient(x_1, axis=0), axis=0)")
                        term_str = term_str.replace('Derivative(x_2, (x, 2))', "np.gradient(np.gradient(x_2, axis=0), axis=0)")
                        
                        term_str = term_str.replace('Derivative(x_1(t, x), (y, 2))', "np.gradient(np.gradient(x_1, axis=1), axis=1)")
                        term_str = term_str.replace('Derivative(x_2(t, x), (y, 2))', "np.gradient(np.gradient(x_2, axis=1), axis=1)")
                        term_str = term_str.replace('Derivative(x_1, (y, 2))', "np.gradient(np.gradient(x_1, axis=1), axis=1)")
                        term_str = term_str.replace('Derivative(x_2, (y, 2))', "np.gradient(np.gradient(x_2, axis=1), axis=1)")
                        
                        term_str = term_str.replace('Derivative(x_1(t, x, y), (x, 2))', "np.gradient(np.gradient(x_1, axis=0), axis=0)")
                        term_str = term_str.replace('Derivative(x_2(t, x, y), (x, 2))', "np.gradient(np.gradient(x_2, axis=0), axis=0)")
                        term_str = term_str.replace('Derivative(x_1(t, x, y), (y, 2))', "np.gradient(np.gradient(x_1, axis=1), axis=1)")
                        term_str = term_str.replace('Derivative(x_2(t, x, y), (y, 2))', "np.gradient(np.gradient(x_2, axis=1), axis=1)")
                        terms.append(f"betas[{j}] * ({term_str})")
                        j += 1

                rhs_str = " + ".join(terms)
                f.write(f"    return {rhs_str}\n\n")

            f.write(f"def system_{n}(X, betas, t):\n")
            f.write(
                "    return np.array([" + ", ".join(f"eq_{n}_{i}(X, betas, t)" for i in range(len(system))) + "])\n\n")