# -*- coding: utf-8 -*-
"""
recreating the plume description in Magorrian Wells 2016
as practice for creating my own with frazil ice
"""

import numpy as np
import math
from scipy.integrate import odeint
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
S_a = .035 #ambient water salinity
#I don't know rho_a, rho_l, T_a, S_i
rho_a = 1.03
rho_l = 1.02
T_a = 0
S_i = 0.035

"""
functions which produce various other (generally non-constant)
values used in the system of differential equations
"""
def get_T(y):
    return T_m + tau * (get_S(y) - S_s) + y[3] / y[0]

def get_T_L(S):
    return T_m + lamda - tau * (S - S_s)

def get_S(y):
    temp = beta_s * S_a + beta_T * (T_m + tau * S_s + y[3] / y[0]) - y[2] / y[0]
    return temp / (beta_s + beta_T * tau)

def get_a(y):
    T_L_S_s = get_T_L(S_s)
    return 1 - c_s / L * (T_s - T_L_S_s)

def get_b(y):
    S = get_S(y)
    print("S " + str(S))
    T_L_S = get_T_L(S)
    print("T_L_S " + str(T_L_S))
    T = get_T(y)
    print("T " + str(T))
    T_L_S_s = get_T_L(S_s)
    return St_m * (1 - c_s / L * (T_s - T_L_S) - St * c_l / L * (T - T_L_S_s))

def get_c(y):
    T = get_T(y)
    print("T " + str(T))
    T_L_S = get_T_L(get_S(y))
    print("T_L_S " + str(T_L_S))
    return - St_m * St * c_l / L * (T - T_L_S)

def get_M(y):
    a = get_a(y)
    print("a " + str(a))
    b = get_b(y)
    print("b " + str(b))
    c = get_c(y)
    print("c " + str(c))
    return (math.sqrt(b ** 2 - 4 * a * c) - b) / (2 * a)

def get_U(y):
    return y[1] / y[0]

def get_H(y):
    return y[0] ** 2 / y[1]

def get_del_rho(y):
    return rho_l * y[2] / y[0]

def get_T_eff():
    T_i = get_T_L(S_i)
    print("T_i")
    print(T_i)
    return T_i - L / c_l + c_s / c_l * (T_s - T_i)

def get_rho_eff():
    print("S part")
    print(beta_s * (S_a - S_s))
    print("T eff")
    print(get_T_eff())
    print("T part")
    print(beta_T * (T_a - get_T_eff()))
    return rho_l * (beta_s * (S_a - S_s) - beta_T * (T_a - get_T_eff()))

def get_del_T_a():
    return T_a - get_T_L(S_a)

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
    del_rho_eff = get_rho_eff()
    M = get_M(y)
    return del_rho_eff / rho_l * M * abs(U)

def dy3_ds(y):
    E = E_0 * sin_theta
    U = get_U(y)
    del_T_a = get_del_T_a()
    M = get_M(y)
    T_eff = get_T_eff()
    S_i = get_S(y)
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
X0 = 0.01
E0 = E_0 * sin_theta
M0 = .0005 * E0
H0 = 2 / 3 * (E0 + M0) * X0
del_T0 = (get_del_T_a() * E0 + get_T_eff() * M0) / (E0 + M0)
del_rho0 = M0 / (E0 + M0) * get_rho_eff()
print(del_rho0)
print(get_rho_eff())
U0 = math.sqrt(2 * (E0 + M0) / (3 * C_d + 4 * (E0 + M0))) * math.sqrt(abs(del_rho0) / rho_l * g * sin_theta * X0)

y0_0 = H0 * U0
y0_1 = H0 * U0 ** 2
y0_2 = H0 * U0 * del_rho0 / rho_l
y0_3 = H0 * U0 * del_T0
#puts these initial values in a vector for solving equations
y0 = [y0_0, y0_1, y0_2, y0_3]

#defines distances along slope to record results
s = np.linspace(0, 100)
y = odeint(derivative, y0, s)

plt.plot(s, y)
plt.show()