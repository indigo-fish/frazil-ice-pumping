# -*- coding: utf-8 -*-
"""
revision of magorrian_wells_odes.py
incorporating frazil ice
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
lamda = 7.61e-4 #depth dependence of freezing point
#lamda = 0
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
#S_a = 35 #ambient water salinity - specified in get_S_a(z) instead
#I don't know rho_a, rho_l, rho_s
rho_s = 916.8
#T_a = 1 #ambient water temperature - specified in get_T_a(z) instead
D = -200 #start depth
U_T = 0 #tidal velocity contributing to drag

#provides linear structure of density
rho_l = 1024

"""
functions which can be changed to specify different
temperature/salinity stratifications
"""
def get_T_a(z):
    T_a = 1 + z / 100
    return T_a

def get_S_a(z):
    S_a = 36
    return S_a

def get_rho_a(z):
    S_a = get_S_a(z)
    T_a = get_T_a(z)
    return 1000 + rho_l * (beta_s * S_a + beta_T * (4 - T_a))

"""
functions which produce various other (generally non-constant)
values used in the system of differential equations
"""
def get_T(y):
    return y[2] / y[0]

def get_T_L(S, z):
    return T_m + lamda * z - tau * (S - S_s)

def get_S(y):
    return y[3] / y[0]

def get_a(z):
    T_L_S_s = get_T_L(S_s, z)
    return 1 - c_s / L * (T_s - T_L_S_s)

def get_b(T, S, z):
    T_L_S = get_T_L(S, z)
    T_L_S_s = get_T_L(S_s, z)
    return St_m * (1 - c_s / L * (T_s - T_L_S) - St * c_l / L * (T - T_L_S_s))

def get_c(T, S, z):
    T_L_S = get_T_L(S, z)
    return - St_m * St * c_l / L * (T - T_L_S)

def get_M(T, S, z):
    a = get_a(z)
    b = get_b(T, S, z)
    c = get_c(T, S, z)
    #currently taking absolute value to make code run
    result = (- b + math.sqrt(abs(b ** 2 - 4 * a * c))) / (2 * a)
    return result

def get_U(y):
    return y[1] / y[0]

def get_H(y):
    return y[0] ** 2 / y[1]

def get_del_rho(y):
    return rho_l * y[2] / y[0]

def get_T_i(y, z):
    S_a = get_S_a(z)
    T_i, S_i = fsolve(solve_system, [get_T_L(S_a, z), S_a], args = (get_M(get_T(y), get_S(y), z), get_U(y), get_T(y), get_S(y), z))
    return T_i

def get_S_i(y, z):
    S_a = get_S_a(z)
    T_i, S_i = fsolve(solve_system, [get_T_L(S_a, z), S_a], args = (get_M(get_T(y), get_S(y), z), get_U(y), get_T(y), get_S(y), z))
    return S_i

def get_del_T_a(z):
    T_a = get_T_a(z)
    S_a = get_S_a(z)
    return T_a - get_T_L(S_a, z)

#need to figure out how to parameterize precipitation of frazil!
def get_p(y, z):
    return 0

#need to figure out how to get phi!
def get_phi(y, z):
    return 0

#system of equations which is solved to find T_i and S_i, given M, U, T and S
def solve_system(vect, M, U, T, S, z):
    T_i, S_i = vect
    func1 = rho_l * L * M * abs(U) + rho_s * c_s * (T_i - T_s) * M * abs(U) - rho_l * c_l * St * abs(U) * (T - T_i)
    func2 = get_T_L(S_i, z) - T_i
    return [func1, func2]

"""
defines each of the linear differential equations
in terms of the variables which make it easier to solve
"""
def dy0_ds(y, z):
    E = E_0 * sin_theta
    U = get_U(y)
    M = get_M(get_T(y), get_S(y), z)
    #need to figure out how to parameterize precipitation
    p = get_p(y, z)
    return E * abs(U) + M * abs(U) + p

def dy1_ds(y, z):
    H = get_H(y)
    U = get_U(y)
    phi = get_phi(y, z)
    return g * sin_theta * H * phi * (1 - rho_s / rho_l) - C_d * U * math.sqrt(U ** 2 + U_T ** 2)

def dy2_ds(y, z):
    E = E_0 * sin_theta
    U = get_U(y)
    T_a = get_T_a(z)
    M = get_M(get_T(y), get_S(y), z)
    p = get_p(y, z)
    result = E * abs(U) * T_a + M * abs(U) * T_s + c_s / c_l * rho_l / rho_s * p * T_s
    return result

def dy3_ds(y, z):
    E = E_0 * sin_theta
    U = get_U(y)
    S = get_S(y)
    return E * abs(U) * S

"""
re-expresses system of differential equations as a vector
"""
def derivative(y, s):
    z = D + s * sin_theta
    dy0 = dy0_ds(y, z)
    dy1 = dy1_ds(y, z)
    dy2 = dy2_ds(y, z)
    dy3 = dy3_ds(y, z)
    return [dy0, dy1, dy2, dy3]

"""
comes up with initial values using analytical solutions
"""

"""
system of equations which is solved to find M, T, S
(under assumption that M is small, so do not need to calculate T_i, S_i
for calculation of initial y values
"""
#need to change this because no longer have analytic solutions
def solve_init_system(vect, E, z):
    M, T, S = vect
    S_a = get_S_a(z)
    T_a = get_T_a(z)
    #func1 is equation 25 in Magorrian Wells
    func1 = get_M(T, S, z) - M
    #func2 is small-M version of equation 29 in Magorrian Wells
    func2 = T - T_m - lamda * z + tau * (S - S_s) - get_del_T_a(z) * E / (E + M)
    #func3 is small-M version of equation 30 in Magorrian Wells
    func3 = beta_s * (S - S_a) - beta_T * (T - T_a)
    return [func1, func2, func3]

X0 = 0.01
E0 = E_0 * sin_theta
z0 = D + X0 * sin_theta

#solves (simplified) analytic system of equations for M, T, S
S_a0 = get_S_a(z0)
T_a0 = get_T_a(z0)
M0, T0, S0 = fsolve(solve_init_system, [E0 * St / (E0 + St) * c_l * get_del_T_a(z0) / L, T_a0, S_a0], args = (E0, z0))

"""
converts M, T, S values into values for U, rho, T_i, S_i
allowing use of full versions of equations 29, 30
"""
H0 = 2 / 3 * (E0 + M0) * X0
del_T0 = T0 - get_T_L(S0, z0)
del_rho0 = - rho_l * (beta_s * (S0 - S_a0) - beta_T * (T0 - T_a0))
U0 = math.sqrt(2 * (E0 + M0 / (3 * C_d + 4 * (E0 + M0)))) * math.sqrt(del_rho0 / rho_l * g * sin_theta * X0)
T_i0, S_i0 = fsolve(solve_system, [get_T_L(S_a0, z0), S_a0], args = (M0, U0, T0, S0, z0))

"""
iterates through more complete version of analytic equations
to find more accurate initial conditions
"""

#converts values into format most useful for differential equations
y0_0 = H0 * U0
y0_1 = H0 * U0 ** 2
y0_2 = H0 * U0 * del_rho0 / rho_l
y0_3 = H0 * U0 * del_T0

#puts these initial values in a vector for solving equations
y0 = [y0_0, y0_1, y0_2, y0_3]


s = np.linspace(0, 100)
H_analytic = []
U_analytic = []
del_T_analytic = []
del_rho_analytic = []

#defines distances along slope to record results
y = odeint(derivative, y0, s)

H_diff = []
U_diff = []
del_T_diff = []
del_S_diff = []

for vect, s0 in zip(y, s):
    z = D + s0 * sin_theta
    H_diff.append(get_H(vect))
    U_diff.append(get_U(vect))
    del_T_diff.append(get_T(vect) - get_T_L(get_S(vect), z))
    del_S_diff.append(get_S(vect) - get_S_a(z))
labels_diff = ["H_diff", "U_diff", "del_T_diff", "del_S_diff"]

array_diff = np.array([H_diff, U_diff, del_T_diff, del_S_diff])

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

for line_diff, label_d in zip(array_diff, labels_diff):
    plt.figure(fig_num)
    plt.plot(s, line_diff, label=label_d)
    plt.legend()
    plt.show()
    fig_num += 1