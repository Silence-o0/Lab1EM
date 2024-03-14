from sympy.plotting import plot
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, fsolve
from sklearn.metrics import r2_score
from sympy import *


def demand_function(x, a, b, c):
    return a * np.exp(-b * x) + c


def supply_function(x, a, b, c):
    return a + b * np.log(c * x + 1)


def equations(variable):
    eq1 = 9.48106745790962 + 109.169046702949 * np.exp(-1.90151984955408 * variable[0])
    eq2 = 0.4336845 + 255.98520268 * np.log(0.3457538 * variable[0] + 1)
    return [eq1 - eq2]


def taxed_equations(variable):
    eq1 = 9.48106745790962 + 109.169046702949 * np.exp(-1.90151984955408 * (variable[0] + 0.1))
    eq2 = 0.4336845 + 255.98520268 * np.log(0.3457538 * variable[0] + 1)
    return [eq1 - eq2]


def arc_elasticity(Q1, Q2, P1, P2):
    arc = ((Q2-Q1)/(P2-P1))/((Q2+Q1)/(P2+P1))
    return arc


# def taxed_demand_function(x, a, b, c, t):
#     return a * np.exp(-b * (x - t)) + c


if __name__ == '__main__':
    price = [0.1, 0.3, 0.45, 0.7, 0.8, 1.05, 1.2, 1.25, 1.31, 1.4, 1.47, 1.55]
    demand = [100, 69, 58, 40, 35, 20, 18, 17, 19, 21, 18, 15]
    supply = [10, 25, 39, 52, 60, 84, 91, 95, 97, 100, 105, 108]

    g = np.linspace(0.1, 1.55, 51)

    init_printing(use_unicode=False, wrap_line=False)
    P = Symbol('P')
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')

    plt.xlabel('Q')
    plt.ylabel('P')

    # Finding optimal parameters for functions
    popt_demand, pcov_demand = curve_fit(demand_function, price, demand)
    popt_supply, pcov_supply = curve_fit(supply_function, price, supply)

    demand_func = a * exp(-b * P) + c
    demand_func_res = demand_func.subs({P: P,
                                        a: popt_demand[0], b: popt_demand[1], c: popt_demand[2]})
    supply_func = a + b * ln(c * P + 1)
    supply_func_res = supply_func.subs({P: P,
                                        a: popt_supply[0], b: popt_supply[1], c: popt_supply[2]})
    print("Demand function:", "Q = ", demand_func, "  ->   ", demand_func_res)
    print("Supply function:", "Q = ", supply_func, "  ->   ", supply_func_res)

    # Draw demand and supply curves
    plt.plot(demand_function(g, popt_demand[0], popt_demand[1], popt_demand[2]), g, label='Demand')
    plt.plot(supply_function(g, popt_supply[0], popt_supply[1], popt_supply[2]), g, label='Supply')

    # Calculate R**2, the coefficient of determination, measuring the degree of
    # agreement between actual and predicted values in a model.
    Q_pred_demand = demand_function(np.array(price), *popt_demand)
    r2_demand = r2_score(demand, Q_pred_demand)
    Q_pred_supply = supply_function(np.array(price), *popt_supply)
    r2_supply = r2_score(supply, Q_pred_supply)
    print("R^2 of demand function:", r2_demand)
    print("R^2 of supply function:", r2_supply)
    print()

    # Check properties
    print("Demand derivatives:")
    demand_first_derivative = diff(demand_func_res, P)
    demand_second_derivative = diff(demand_first_derivative, P)
    print("First derivative:", demand_first_derivative, ' < 0')
    print("Second derivative:", demand_second_derivative, ' >= 0')
    print()

    print("Supply derivatives:")
    supply_first_derivative = diff(supply_func_res, P)
    supply_second_derivative = diff(supply_first_derivative, P)
    print("First derivative:", supply_first_derivative, ' > 0')
    print("Second derivative:", supply_second_derivative, ' <= 0')
    print()

    # Draw points
    plt.scatter(demand, price)
    plt.scatter(supply, price)

    # Connect points on diagram
    plt.plot(demand, price)
    plt.plot(supply, price)

    # Finding equilibrium point
    P_eqp = fsolve(equations, 0.6)[0]
    Q_eqp = demand_func_res.subs({P: P_eqp})
    print(f"Equilibrium point: ({round(Q_eqp, 3)}; {round(P_eqp, 3)})")
    plt.plot(Q_eqp, P_eqp, 'bo', label='Equilibrium point')

    #Equilibrium point stability
    E_d_eqp = demand_first_derivative.subs(P, P_eqp)/(Q_eqp/P_eqp)
    print("Point elasticity of demand: ", E_d_eqp)
    E_s_eqp = supply_first_derivative.subs(P, P_eqp)/(Q_eqp/P_eqp)
    print("Point elasticity of supply: ", E_s_eqp)

    if abs(E_d_eqp) > abs(E_s_eqp):
        print("Point is stable")
    else:
        print("Point is not stable")

    #Arc elasticity of demand
    # print(arc_elasticity(demand[3], demand[4], price[3], price[4]))
    # print(arc_elasticity(demand[11], demand[0], price[11], price[0]))

    #After taxation on demand
    t = 0.1
    # taxed_demand_func = a * np.exp(-b * (P + t)) + c
    plt.plot(demand_function(g - g*t, popt_demand[0], popt_demand[1], popt_demand[2]), g, label='Taxed demand')
    P_eqp_taxed = fsolve(taxed_equations, 0.6)[0]
    Q_eqp_taxed = supply_func_res.subs({P: P_eqp_taxed})
    print(f"Equilibrium point: ({round(Q_eqp_taxed, 3)}; {round(P_eqp_taxed, 3)})")
    plt.plot(Q_eqp_taxed, P_eqp_taxed, 'bo', label='Equilibrium point')
    # print(taxed_demand_func)

    plt.legend()
    plt.show()