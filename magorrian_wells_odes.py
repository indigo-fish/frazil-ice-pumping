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
S_a = 35 #ambient water salinity
#I don't know rho_a, rho_l, rho_s
rho_s = 916.8
#T_a is supposed to vary
T_a = -.5 #ambient water temperature

"""
#simplified constants for testing code
T_s = 0 #background ice temperature
E_0 = .1 #max entrainment coefficient
C_d = 1 #drag coefficient
St = 1 #heat transfer coefficient
St_m = 1 #salt transfer coefficient
tau = 0 #seawater freezing point slope
T_m = 0.5 #freezing point offset
lamda = 0 #depth dependence of freezing point
L = 1 #latent heat of fusion for ice
c_s = 1 #specific heat capacity for ice
c_l = 1 #specific heat capacity for seawater
beta_s = 1 #haline contraction coefficient
beta_T = 1 #thermal expansion coefficient
g = 1 #gravitational acceleration
S_s = 0 #salt concentration in ice
#I am choosing theta arbitrarily
theta = 0.1
sin_theta = math.sin(theta)
S_a = 35 #ambient water salinity
#I don't know rho_a, rho_l, rho_s
rho_s = 900
#T_a is supposed to vary
T_a = 0 #ambient water temperature
"""

"""
rho_a = 1000 / (1 + beta_T * (T_a - 4) - beta_s * S_a)
rho_l = rho_a
"""


rho_l = 1024
rho_a = 1000 + rho_l * (beta_s * S_a + beta_T * (4 - T_a))


"""
functions which produce various other (generally non-constant)
values used in the system of differential equations
"""
def get_T(y):
    result = T_m - tau * (get_S(y) - S_s) + y[3] / y[0]
    return result

def get_T_L(S):
    return T_m - tau * (S - S_s)

def get_S(y):
    temp = beta_s * S_a + beta_T * (T_m + tau * S_s + y[3] / y[0]) - y[2] / y[0]
    result = temp / (beta_s + beta_T * tau)
    return result

def get_a():
    T_L_S_s = get_T_L(S_s)
    return 1 - c_s / L * (T_s - T_L_S_s)

def get_b(T, S):
    T_L_S = get_T_L(S)
    T_L_S_s = get_T_L(S_s)
    return St_m * (1 - c_s / L * (T_s - T_L_S) - St * c_l / L * (T - T_L_S_s))

def get_c(T, S):
    T_L_S = get_T_L(S)
    return - St_m * St * c_l / L * (T - T_L_S)

def get_M(T, S):
    
    a = get_a()
    b = get_b(T, S)
    c = get_c(T, S)
    result = (- b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return result

def get_U(y):
    return y[1] / y[0]

def get_H(y):
    return y[0] ** 2 / y[1]

def get_del_rho(y):
    return rho_l * y[2] / y[0]

def get_T_i(y):
    T_i, S_i = fsolve(solve_system, [get_T_L(S_a), S_a], args = (get_M(get_T(y), get_S(y)), get_U(y), get_T(y), get_S(y)))
    return T_i

def get_S_i(y):
    T_i, S_i = fsolve(solve_system, [get_T_L(S_a), S_a], args = (get_M(get_T(y), get_S(y)), get_U(y), get_T(y), get_S(y)))
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
    M = get_M(get_T(y), get_S(y))
    return E * abs(U) + M * abs(U)

def dy1_ds(y):
    H = get_H(y)
    U = get_U(y)
    delta_rho = get_del_rho(y)
    return g * sin_theta * H * delta_rho - C_d * U * abs(U)

def dy2_ds(y):
    U = get_U(y)
    del_rho_eff = get_rho_eff(y)
    M = get_M(get_T(y), get_S(y))
    return del_rho_eff / rho_l * M * abs(U)

def dy3_ds(y):
    E = E_0 * sin_theta
    U = get_U(y)
    del_T_a = get_del_T_a()
    M = get_M(get_T(y), get_S(y))
    T_eff = get_T_eff(get_T_i(y))
    T_L_S_s = get_T_L(S_s)
    derivative = del_T_a * E * abs(U) + M * abs(U) * (T_eff - T_L_S_s) - lamda * sin_theta * y[0]
    return derivative

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
system of equations which is solved to find M, T, S
(under assumption that M is small, so do not need to calculate T_i, S_i
for calculation of initial y values
"""
def solve_init_system(vect, E):
    M, T, S = vect
    #func1 is equation 25 in Magorrian Wells
    func1 = get_M(T, S) - M
    #func2 is small-M version of equation 29 in Magorrian Wells
    func2 = T - T_m - tau * (S - S_s) - get_del_T_a() * E / (E + M)
    #func3 is small-M version of equation 30 in Magorrian Wells
    func3 = beta_s * (S - S_a) - beta_T * (T - T_a)
    return [func1, func2, func3]

X0 = 0.01
E0 = E_0 * sin_theta

#solves (simplified) analytic system of equations for M, T, S
M0, T0, S0 = fsolve(solve_init_system, [E0 * St / (E0 + St) * c_l * get_del_T_a() / L, T_a, S_a], args = (E0))

"""
converts M, T, S values into values for U, rho, T_i, S_i
allowing use of full versions of equations 29, 30
"""
H0 = 2 / 3 * (E0 + M0) * X0
del_T0 = T0 - get_T_L(S0)
del_rho0 = - rho_l * (beta_s * (S0 - S_a) - beta_T * (T0 - T_a))
U0 = math.sqrt(2 * (E0 + M0 / (3 * C_d + 4 * (E0 + M0)))) * math.sqrt(del_rho0 / rho_l * g * sin_theta * X0)
T_i0, S_i0 = fsolve(solve_system, [get_T_L(S_a), S_a], args = (M0, U0, T0, S0))

def repeat_init_solve(vect, E, X, U, T_i, S_i):
    T_eff = get_T_eff(T_i)
    T_L_S_s = get_T_L(S_s)
    del_T_eff = T_eff - T_L_S_s
    del_rho_eff = rho_l * (beta_s * (S_a - S_s) - beta_T * (T_a - T_eff))
    M, T, S = vect
    del_T = T - get_T_L(S)
    del_rho = -1 * rho_l * (beta_s * (S - S_a) - beta_T * (T - T_a))
    #func1 is equation 25 in Magorrian Wells
    func1 = get_M(T, S) - M
    #func2 is equation 29 in Magorrian Wells
    func2 = (E * get_del_T_a() + del_T_eff * M) / (E + M) - del_T
    #func3 is equation 30 in Magorrian Wells
    func3 = M / (E + M) * del_rho_eff - del_rho
    return [func1, func2, func3]

"""
iterates through more complete version of analytic equations
to find more accurate initial conditions
"""
i_vals = []
del_T_vals = []
del_rho_vals = []
U_vals = []
M_vals = []
i = 0
while i < 40:
    M0, T0, S0 = fsolve(repeat_init_solve, [M0, T0, S0], args = (E0, X0, U0, T_i0, S_i0))
    H0 = 2 / 3 * (E0 + M0) * X0
    del_T0 = T0 - get_T_L(S0)
    del_rho0 = - rho_l * (beta_s * (S0 - S_a) - beta_T * (T0 - T_a))
    U0 = math.sqrt(2 * (E0 + M0 / (3 * C_d + 4 * (E0 + M0)))) * math.sqrt(del_rho0 / rho_l * g * sin_theta * X0)
    T_i0, S_i0 = fsolve(solve_system, [get_T_L(S_a), S_a], args = (M0, U0, T0, S0))
    i_vals.append(i)
    del_T_vals.append(del_T0)
    del_rho_vals.append(del_rho0)
    U_vals.append(U0)
    M_vals.append(M0)
    i += 1

# print("del_T0 " + str(del_T0))
# print("del_rho0 " + str(del_rho0))
# print("U0 " + str(U0))
# print("H0 " + str(H0))
# print("M0 " + str(M0))

#functions for calculating analytic values at locations other than initial point
def analytic_H(E, M, X):
    return 2 / 3 * (E + M) * X

def analytic_U(E, M, X, T_i):
    return math.sqrt(2 * (E + M) / (3 * C_d + 4 * (E + M))) * math.sqrt(analytic_del_rho(E, M, X, T_i) * g * sin_theta * X)

def analytic_del_T(E, M, X):
    return (get_del_T_a() * E + (T_i0 - get_T_L(S_s)) * M) / (E + M)

def analytic_del_rho(E, M, X, T_i):
    T_eff = get_T_eff(T_i)
    del_rho_eff = rho_l * (beta_s * (S_a - S_s) - beta_T * (T_a - T_eff))
    return M / (E + M) * rho_l * del_rho_eff

#converts analytic functions into vector
def analytic_values(X, E, M, T_i):
    H = analytic_H(E, M, X)
    U = analytic_U(E, M, X, T_i)
    del_T = analytic_del_T(E, M, X)
    del_rho = analytic_del_rho(E, M, X, T_i)
    return [H, U, del_T, del_rho]

s = np.linspace(0, 100)
H_analytic = []
U_analytic = []
del_T_analytic = []
del_rho_analytic = []
#evaluates analytic functions at all points reported for differential equation solutions
for s0 in s:
    an_val = analytic_values(s0, E0, M0, T_i0)
    H_analytic.append(an_val[0])
    U_analytic.append(an_val[1])
    del_T_analytic.append(an_val[2])
    del_rho_analytic.append(an_val[3])
labels_analytic = ["H_analytic", "U_analytic", "del_T_analytic", "del_rho_analytic"]

array_analytic = np.array([H_analytic, U_analytic, del_T_analytic, del_rho_analytic])

#converts solved-for values into initial values in y
H0 = 2 / 3 * (E0 + M0) * X0
del_T0 = T0 - get_T_L(S0)
del_rho0 = - rho_l * (beta_s * (S0 - S_a) - beta_T * (T0 - T_a))
U0 = math.sqrt(2 * (E0 + M0 / (3 * C_d + 4 * (E0 + M0)))) * math.sqrt(del_rho0 / rho_l * g * sin_theta * X0)

y0_0 = H0 * U0
y0_1 = H0 * U0 ** 2
y0_2 = H0 * U0 * del_rho0 / rho_l
y0_3 = H0 * U0 * del_T0

#puts these initial values in a vector for solving equations
y0 = [y0_0, y0_1, y0_2, y0_3]
print(y0)

#defines distances along slope to record results
y = odeint(derivative, y0, s)

H_diff = []
U_diff = []
del_T_diff = []
del_rho_diff = []

for vect in y:
    H_diff.append(get_H(vect))
    U_diff.append(get_U(vect))
    del_T_diff.append(get_T(vect) - get_T_L(get_S(vect)))
    del_rho_diff.append(get_del_rho(vect))
labels_diff = ["H_diff", "U_diff", "del_T_diff", "del_rho_diff"]

array_diff = np.array([H_diff, U_diff, del_T_diff, del_rho_diff])

# print("del_T0 " + str(analytic_del_T(E0, M0, X0)))
# print("del_rho0 " + str(analytic_del_rho(E0, M0, X0)))
# print("U0 " + str(analytic_U(E0, M0, X0)))
# print("H0 " + str(analytic_H(E0, M0, X0)))

#plots first differential equations and second analytic solutions
"""
currently there is barely any agreement at all
between the two solutions, which should agree
"""

fig_num = 1

for line_diff, line_analytic, label_d, label_a in zip(array_diff, array_analytic, labels_diff, labels_analytic):
    plt.figure(fig_num)
    plt.scatter(s, line_diff, label=label_d)
    plt.scatter(s, line_analytic, label=label_a)
    plt.legend()
    plt.show()
    fig_num += 1

plt.figure(fig_num)
plt.scatter(i_vals, del_T_vals, label="del_T")
plt.scatter(i_vals, del_rho_vals, label="del_rho")
plt.scatter(i_vals, U_vals, label="U")
plt.scatter(i_vals, M_vals, label="M")
plt.legend()
plt.show()