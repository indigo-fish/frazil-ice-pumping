# -*- coding: utf-8 -*-
"""
revision of magorrian_wells_odes.py
incorporating frazil ice
with differential equations in terms of delta T and delta rho
"""

import numpy as np
import math
from scipy.integrate import odeint
from scipy.integrate import quad
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
#lamda = 7.61e-4 #depth dependence of freezing point
lamda = 0
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
nu = 1.95e-6
#T_a = 1 #ambient water temperature - specified in get_T_a(z) instead
D = -200 #start depth
U_T = 0

#provides linear structure of density
rho_l = 1024

"""
functions which can be changed to specify different
temperature/salinity stratifications
"""
def get_T_a(z):
    integral, error = quad(get_d_T_a_dz, D, z)
    T_a = 0 + integral
    #T_a = 0
    return T_a

def get_S_a(z):
    integral, error = quad(get_d_S_a_dz, D, z)
    S_a = 35 + integral
    return S_a

def get_rho_a(z):
    S_a = get_S_a(z)
    T_a = get_T_a(z)
    return 1000 + rho_l * (beta_s * S_a + beta_T * (4 - T_a))

def get_d_T_a_dz(z):
    return 0

def get_d_S_a_dz(z):
    return 0

def get_d_rho_a_dz(z):
    return rho_l * (beta_s * get_d_S_a_dz(z) - beta_T * get_d_T_a_dz(z))

"""
functions which produce various other (generally non-constant)
values used in the system of differential equations
"""
def get_T(y, z):
    result = T_m + lamda * z - tau * (get_S(y, z) - S_s) + y[3] / y[0]
    return result

def get_T_L(S, z):
    return T_m + lamda * z - tau * (S - S_s)

def get_S(y, z):
    S_a = get_S_a(z)
    temp = beta_s * S_a + beta_T * (T_m + lamda * z + tau * S_s + y[3] / y[0]) - y[2] / y[0]
    result = temp / (beta_s + beta_T * tau)
    return result

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
    #temporarily taking absolute value to make code run
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
    T_i, S_i = fsolve(solve_system, [get_T_L(S_a, z), S_a], args = (get_M(get_T(y, z), get_S(y, z), z), get_U(y), get_T(y, z), get_S(y, z), z))
    return T_i

def get_S_i(y, z):
    S_a = get_S_a(z)
    T_i, S_i = fsolve(solve_system, [get_T_L(S_a, z), S_a], args = (get_M(get_T(y, z), get_S(y, z), z), get_U(y), get_T(y, z), get_S(y, z), z))
    return S_i

def get_T_eff(T_i):
    return T_i - L / c_l + c_s / c_l * (T_s - T_i)

def get_rho_eff(y, z):
    S_a = get_S_a(z)
    T_a = get_T_a(z)
    return rho_l * (beta_s * (S_a - S_s) - beta_T * (T_a - get_T_eff(get_T_i(y, z))))

def get_del_T_a(z):
    T_a = get_T_a(z)
    S_a = get_S_a(z)
    return T_a - get_T_L(S_a, z)

#calculates precipitation rate assuming Stokes drag/buoyancy balance and spherical ice crystals
def get_p(y, z):
    R = get_radius(y)
    v_rel = 2 / 9 * R ** 2 * g / (nu * rho_l) #vertical ice velocity relative to surrounding liquid
    phi = get_phi(y, z)
    return rho_s / rho_l * phi * v_rel

#need to figure out how to choose average radius of ice crystals!
def get_radius(y):
    return .5e-3

#calculates phi by dividing U * phi / U
def get_phi(y, z):
    return y[4] / get_U(y)

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
#derivative of HU
def dy0_ds(y, z):
    U = get_U(y)
    e = E_0 * sin_theta * abs(U)
    m = get_M(get_T(y, z), get_S(y, z), z) * abs(U)
    #need to figure out how to parameterize precipitation
    p = get_p(y, z)
    return e + m + p

#derivative of HU^2
def dy1_ds(y, z):
    H = get_H(y)
    U = get_U(y)
    phi = get_phi(y, z)
    delta_rho = get_del_rho(y)
    result = g * sin_theta * H * (phi * (1 - rho_s / rho_l) + delta_rho / rho_l) - C_d * U * math.sqrt(U ** 2 + U_T ** 2)
    return result

def dy2_ds(y, z):
    U = get_U(y)
    del_rho_eff = get_rho_eff(y, z)
    m = get_M(get_T(y, z), get_S(y, z), z) * abs(U)
    drho_a_ds = get_d_rho_a_dz(z) * sin_theta
    S_a = get_S_a(z)
    T_a = get_T_a(z)
    T_i = get_T_i(y, z)
    p = get_p(y, z)
    temp = drho_a_ds / rho_l * y[0] + del_rho_eff / rho_l * m
    #need to change factors of rho to account for specific, not volume, heat capacity
    result = temp + (beta_s * S_a - beta_T * (T_a - c_s / c_l * T_i)) * p + beta_T * rho_s / rho_l * L / c_l * dy4_ds(y, z)
    return result

def dy3_ds(y, z):
    U = get_U(y)
    e = E_0 * sin_theta * abs(U)
    del_T_a = get_del_T_a(z)
    m = get_M(get_T(y, z), get_S(y, z), z) * abs(U)
    T_i = get_T_i(y, z)
    T_eff = get_T_eff(T_i)
    T_L_S_s = get_T_L(S_s, z)
    p = get_p(y, z)
    temp = del_T_a * e + m * (T_eff - T_L_S_s) - lamda * sin_theta * y[0]
    #need to change factors of rho to account for specific, not volume, heat capacity
    result = temp + (c_s / c_l * T_i - (L / c_l + T_m + lamda * z)) * p + rho_s / rho_l * L / c_l * dy4_ds(y, z)
    return result

def dy4_ds(y, z):
    """
    T = get_T(y)
    S = get_S(y)
    T_L = get_T_L(S, z)
    if T > T_L: result = 0
    else:
        T_i = get_T_i(y, z)
        U = get_U(y)
        p = get_p(y, z)
        e = E_0 * sin_theta * abs(U)
        m = get_M(T, S, z) * abs(U)
        T_a = get_T_a(z)
        gamma_T = get_gamma_T(y)
        temp = (T_m + lamda * z) * dy0_ds(y, z) - tau * dy3_ds(y, z) + lamda * y[0] * sin_theta - e * T_a - m * T_i - c_s / c_l * p * T_i + gamma_T * (T - T_i)
        result = (temp * rho_l * c_l / L + rho_l * p) / rho_s
    """
    result = 0
    return result

"""
re-expresses system of differential equations as a vector
"""
def derivative(y, s):
    z = D + s * sin_theta
    dy0 = dy0_ds(y, z)
    dy1 = dy1_ds(y, z)
    dy2 = dy2_ds(y, z)
    dy3 = dy3_ds(y, z)
    dy4 = dy4_ds(y, z)
    U = get_U(y)
    #e = E_0 * sin_theta * abs(U)
    T = get_T(y, z)
    S = get_S(y, z)
    #m = get_M(T, S, z) * abs(U)
    #T_i = get_T_i(y, z)
    #alt_form_del_T = e * get_T_a(z) + m * T_i - St * abs(U) * (T - T_i)
    #print(alt_form_del_T)
    return [dy0, dy1, dy2, dy3, dy4]

"""
comes up with initial values using analytical solutions
"""

"""
system of equations which is solved to find M, T, S
(under assumption that M is small, so do not need to calculate T_i, S_i
for calculation of initial y values
"""
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

def repeat_init_solve(vect, E, X, U, T_i, S_i):
    z = D + X * sin_theta
    T_eff = get_T_eff(T_i)
    T_L_S_s = get_T_L(S_s, z)
    del_T_eff = T_eff - T_L_S_s
    S_a = get_S_a(z)
    T_a = get_T_a(z)
    del_rho_eff = rho_l * (beta_s * (S_a - S_s) - beta_T * (T_a - T_eff))
    M, T, S = vect
    del_T = T - get_T_L(S, z)
    del_rho = -1 * rho_l * (beta_s * (S - S_a) - beta_T * (T - T_a))
    #func1 is equation 25 in Magorrian Wells
    func1 = get_M(T, S, z) - M
    #func2 is equation 29 in Magorrian Wells
    func2 = (E * get_del_T_a(z) + del_T_eff * M) / (E + M) - del_T
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
    #converts output of system of equations into other values
    #some of which are in turn inputs in next iteration of system of equations
    H0 = 2 / 3 * (E0 + M0) * X0
    del_T0 = T0 - get_T_L(S0, z0)
    del_rho0 = -1 * rho_l * (beta_s * (S0 - S_a0) - beta_T * (T0 - T_a0))
    T_eff = get_T_eff(T_i0)
    rho_eff = rho_l * (beta_s * (S_a0 - S_s) - beta_T * (T_a0 - get_T_eff(T_i0)))
    U0 = math.sqrt(2 * (E0 + M0 / (3 * C_d + 4 * (E0 + M0)))) * math.sqrt(del_rho0 / rho_l * g * sin_theta * X0)
    T_i0, S_i0 = fsolve(solve_system, [get_T_L(S_a0, z0), S_a0], args = (M0, U0, T0, S0, z0))
    i_vals.append(i)
    del_T_vals.append(del_T0)
    del_rho_vals.append(del_rho0)
    U_vals.append(U0)
    M_vals.append(M0)
    i += 1

#converts values into format most useful for differential equations
y0_0 = H0 * U0
y0_1 = H0 * U0 ** 2
y0_2 = H0 * U0 * T0
y0_3 = H0 * U0 * S0
y0_4 = 0

print(T0)
#puts these initial values in a vector for solving equations
y0 = [y0_0, y0_1, y0_2, y0_3, y0_4]
print(y0)

#functions for calculating analytic values at locations other than initial point
def analytic_H(E, M, X):
    return 2 / 3 * (E + M) * X

def analytic_U(E, M, X, T_i):
    return math.sqrt(2 * (E + M) / (3 * C_d + 4 * (E + M))) * math.sqrt(analytic_del_rho(E, M, X, T_i) / rho_l * g * sin_theta * X)

def analytic_del_T(E, M, X):
    z = D + X * sin_theta
    del_T_a = get_del_T_a(z)
    #T_a = T_a0
    result = (del_T_a * E + (get_T_eff(T_i0) - get_T_L(S_s, z)) * M) / (E + M)
    return result

def analytic_del_rho(E, M, X, T_i):
    z = D + X * sin_theta
    S_a = get_S_a(z)
    T_a = get_T_a(z)
    #S_a = S_a0
    #T_a = T_a0
    T_eff = get_T_eff(T_i)
    del_rho_eff = rho_l * (beta_s * (S_a - S_s) - beta_T * (T_a - T_eff))
    return M / (E + M) * del_rho_eff

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

#defines distances along slope to record results
y = odeint(derivative, y0, s)

H_diff = []
U_diff = []
del_T_diff = []
del_rho_diff = []
phi_diff = []
del_S_diff = []
T_diff = []
T_L_diff = []

for vect, s0 in zip(y, s):
    z = D + s0 * sin_theta
    H_diff.append(get_H(vect))
    U_diff.append(get_U(vect))
    del_T_diff.append(get_T(vect, z) - get_T_L(get_S(vect, z), z))
    del_rho_diff.append(get_del_rho(vect))
    phi_diff.append(get_phi(vect, z))
    del_S_diff.append(get_S(vect, z) - get_S_a(z))
    T_diff.append(get_T(vect, z))
    T_L_diff.append(get_T_L(get_S(vect, z), z))
labels_diff = ["H_diff", "U_diff", "del_T_diff", "del_rho_diff"]
labels_diff_2 = ["del_S_diff", "T_diff", "T_L_diff"]

array_diff = np.array([H_diff, U_diff, del_T_diff, del_rho_diff])
array_diff_2 = np.array([del_S_diff, T_diff, T_L_diff])

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
    plt.plot(s, line_diff, label=label_d)
    plt.plot(s, line_analytic, label=label_a)
    plt.legend()
    plt.show()
    fig_num += 1

for line_diff, label_d in zip(array_diff_2, labels_diff_2):
    plt.figure(fig_num)
    plt.plot(s, line_diff, label=label_d)
    plt.legend()
    plt.show()
    fig_num += 1