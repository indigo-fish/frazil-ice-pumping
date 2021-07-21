# -*- coding: utf-8 -*-
"""
recreating the plume description in Magorrian Wells 2016
as practice for creating my own with frazil ice
"""

import numpy as np
import math
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

"""
defines all the constants that will be
used throughout various functions and formulas
"""

T_s = -5 #background ice temperature
E_0 = 3.6e-2 #max entrainment coefficient
C_d = 2.5e-3 #drag coefficient
St = 1.1e-3 #heat transfer coefficient
St_m = 3.1e-5 #salt transfer coefficient
tau = 5.73e-2 #seawater freezing point slope
T_m = 8.32e-2 #freezing point offset
lamda = 0 #depth dependence of freezing point
L = 3.35e5 #latent heat of fusion for ice
c_s = 2.009e3 #specific heat capacity for ice
c_l = 3.974e3 #specific heat capacity for seawater
beta_s = 7.86e-4 #haline contraction coefficient
beta_T = 3.87e-5 #thermal expansion coefficient
g = 9.81 #gravitational acceleration
S_s = 0 #salt concentration in ice
#I am choosing theta arbitrarily
theta = 0.1
sin_theta = math.sin(theta)
S_a = 34 #ambient water salinity
#I don't know rho_a, rho_l, rho_s
rho_a = 1023.6
rho_l = 1023.6
rho_s = 916.8
#T_a is supposed to vary
T_a = 0

"""
functions which produce various other (generally non-constant)
values used in the system of differential equations
"""
def get_T(y):
    return T_m - tau * (get_S(y) - S_s) + y[3] / y[0]

def get_T_L(S):
    return T_m - tau * (S - S_s)

def get_S(y):
    temp = beta_s * S_a + beta_T * (T_m + tau * S_s + y[3] / y[0]) - y[2] / y[0]
    return temp / (beta_s + beta_T * tau)

def get_a(y):
    T_L_S_s = get_T_L(S_s)
    return 1 - c_s / L * (T_s - T_L_S_s)

def get_b(y):
    S = get_S(y)
    T_L_S = get_T_L(S)
    T = get_T(y)
    T_L_S_s = get_T_L(S_s)
    return St_m * (1 - c_s / L * (T_s - T_L_S) - St * c_l / L * (T - T_L_S_s))

def get_c(y):
    T = get_T(y)
    T_L_S = get_T_L(get_S(y))
    return - St_m * St * c_l / L * (T - T_L_S)

def get_M(y):
    a = get_a(y)
    print(a)
    b = get_b(y)
    print(b)
    c = get_c(y)
    print(c)
    return (math.sqrt(b ** 2 - 4 * a * c) - b) / (2 * a)

def get_U(y):
    return y[1] / y[0]

def get_H(y):
    return y[0] ** 2 / y[1]

def get_del_rho(y):
    return rho_l * y[2] / y[0]

def get_T_i(y):
    T_i, S_i = fsolve(solve_system, [get_T_L(S_a), S_a], args = (get_M(y), get_U(y), get_T(y), get_S(y)))
    return T_i

def get_S_i(y):
    T_i, S_i = fsolve(solve_system, [get_T_L(S_a), S_a], args = (get_M(y), get_U(y), get_T(y), get_S(y)))
    return S_i

def get_T_eff(T_i):
    return T_i - L / c_l + c_s / c_l * (T_s - T_i)

def get_rho_eff(y):
    return rho_l * (beta_s * (S_a - S_s) - beta_T * (T_a - get_T_eff(get_T_i(y))))

def get_del_T_a():
    return T_a - get_T_L(S_a)

#system of equations which is solved to find T_i and S_i, given M, U, T and S
def solve_system(vect, M, U, T, S):
    T_i, S_i = vect
    func1 = rho_l * L * M * abs(U) + rho_s * c_s * (T_i - T_s) * M * abs(U) - rho_l * c_l * St * abs(U) * (T - T_i)
    func2 = get_T_L(S_i) - T_i
    return [func1, func2]

"""
defines each of the linear differential equations
in terms of the variables which make it easier to solve
"""
def dy0_ds(y):
    E = E_0 * sin_theta
    U = get_U(y)
    M = get_M(y)
    return E * abs(U) + M * abs(U)

def dy1_ds(y):
    H = get_H(y)
    U = get_U(y)
    delta_rho = get_del_rho(y)
    return g * sin_theta * H * delta_rho - C_d * U * abs(U)

def dy2_ds(y):
    U = get_U(y)
    del_rho_eff = get_rho_eff(y)
    M = get_M(y)
    return del_rho_eff / rho_l * M * abs(U)

def dy3_ds(y):
    E = E_0 * sin_theta
    U = get_U(y)
    del_T_a = get_del_T_a()
    M = get_M(y)
    T_eff = get_T_eff(get_T_i(y))
    S_i = get_S_i(y)
    return del_T_a * E * abs(U) + M * abs(U) * (T_eff - T_m + tau * S_i) - lamda * sin_theta * y[0]

"""
re-expresses system of differential equations as a vector
"""
def derivative(y, t):
    dy0 = dy0_ds(y)
    dy1 = dy1_ds(y)
    dy2 = dy2_ds(y)
    dy3 = dy3_ds(y)
    return [dy0, dy1, dy2, dy3]

"""
comes up with initial values using analytical solutions
"""

"""
system of equations which is solved to find M, U, T, S, T_i and S_i
for calculation of initial y values
"""
def solve_init_system(vect, E, X):
    M, U, T, S, T_i, S_i = vect
    func1 = beta_T * (T - T_a) - beta_s * (S - S_a) + M / (E + M) * (beta_s * (S_a - S_s) - beta_T * (T_a - get_T_eff(T_i)))
    func2 = T + T_m + tau * (S - S_s) + ((T_a - get_T_L(S_a)) * E + (T_i - L / c_l + c_s / c_l * (T_s - T_i) - T_m + tau * (S - S_s)) * M) / (E + M)
    func3 = - T_i + T_m - tau * (S - S_s)
    func4, func5 = solve_system([T_i, S_i], M, U, T, S)
    """
    this is currently using absolute value inside square root
    due to M being negative
    """
    func6 = - U + math.sqrt(2 * (E + M) / (3 * C_d + 4 * (E + M))) * math.sqrt(abs(M / (E + M) * (beta_s * (S_a - S_s) - beta_T * (T_a - get_T_eff(T_i))) * g * sin_theta * X))
    return [func1, func2, func3, func4, func5, func6]

X0 = 0.01
E0 = E_0 * sin_theta

#solves analytic system of equations for M, U, T, S, T_i, S_i
M0, U0, T0, S0, T_i0, S_i0 = fsolve(solve_init_system, [.0005 * E0, .01, T_a, S_a, T_a, S_a], args = (E0, X0))

print("M0 " + str(M0))
print("U0 " + str(U0))
print("T0 " + str(T0))
print("S0 " + str(S0))
print("T_i0 " + str(T_i0))
print("S_i0 " + str(S_i0))

#converts solved-for values into initial values in y
H0 = 2 / 3 * (E0 + M0) * X0
del_T0 = T0 - get_T_L(S0)
del_rho0 = - rho_l * (beta_s * (S0 - S_a) - beta_T * (T0 - T_a))

y0_0 = H0 * U0
y0_1 = H0 * U0 ** 2
y0_2 = H0 * U0 * del_rho0 / rho_l
y0_3 = H0 * U0 * del_T0

#puts these initial values in a vector for solving equations
y0 = [y0_0, y0_1, y0_2, y0_3]

#defines distances along slope to record results
s = np.linspace(0, 100)
#y = odeint(derivative, y0, s)

#functions for calculating analytic values at locations other than initial point
def analytic_H(E, M, X):
    return 2 / 3 * (E + M) * X

"""
again using absolute value in square root due to negative M
"""
def analytic_U(E, M, X):
    return math.sqrt(2 * (E + M) / (3 * C_d + 4 * (E + M))) * math.sqrt(abs(M / (E + M) * (beta_s * (S_a - S_s) - beta_T * (T_a - get_T_eff(T_i0))) * g * sin_theta * X))

def analytic_del_T(E, M, X):
    return (get_del_T_a() * E + (T_i0 - get_T_L(S_s)) * M) / (E + M)

def analytic_del_rho(E, M, X):
    return M / (E + M) * rho_l * (beta_s * (S_a - S_s) - beta_T * (T_a - T_i0))

#converts analytic functions into vector
def analytic_values(X):
    H = analytic_H(E0, M0, X)
    U = analytic_U(E0, M0, X)
    del_T = analytic_del_T(E0, M0, X)
    del_rho = analytic_del_rho(E0, M0, X)
    return [H, U, del_T, del_rho]

H_vals = []
U_vals = []
del_T_vals = []
del_rho_vals = []
#evaluates analytic functions at all points reported for differential equation solutions
for s0 in s:
    an_val = analytic_values(s0)
    H_vals.append(an_val[0])
    U_vals.append(an_val[1])
    del_T_vals.append(an_val[2])
    del_rho_vals.append(an_val[3])
analytic_y = [H_vals, U_vals, del_T_vals, del_rho_vals]
labels = ["H", "U", "del_T", "del_rho"]

array_y = np.array(analytic_y)

#plots first differential equations and second analytic solutions
"""
currently there is barely any agreement at all
between the two solutions, which should agree
"""
#plt.figure(1)
#plt.plot(s, y)
#plt.show()

fig_num = 2
for y_line, label in zip(array_y, labels):
    plt.figure(fig_num)
    plt.plot(s, y_line, label=label)
    plt.legend()
    plt.show()
    fig_num += 1