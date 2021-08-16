"""
alternate version of frazil_plume_del_rho.py
using only dominant terms in differential equations
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
lamda = 7.61e-4 #depth dependence of freezing point
L = 3.35e5 #latent heat of fusion for ice
c_s = 2.009e3 #specific heat capacity for ice
c_l = 3.974e3 #specific heat capacity for seawater
beta_s = 7.86e-4 #haline contraction coefficient
beta_T = 3.87e-5 #thermal expansion coefficient
g = 9.81 #gravitational acceleration
S_s = 0 #salt concentration in ice
#I am choosing theta arbitrarily
#theta = 0.1
sin_theta = 1115/600e3
#S_a = 35 #ambient water salinity - specified in get_S_a(z) instead
#I don't know rho_a, rho_l, rho_s
rho_s = 916.8
nu = 1.95e-6
#T_a = 1 #ambient water temperature - specified in get_T_a(z) instead
D = -1400 #start depth
U_T = 0 #tide velocity
get_R = 5e-3 #global variable for radius set by later code
my_precipitation = True #chooses between Stokes and Jenkins drag
Jenkins_ambient = True #chooses between ideal and Amery ice shelf conditions

#provides linear structure of density
rho_l = 1028

"""
functions which can be changed to specify different
temperature/salinity stratifications
"""
#integrates derivative to find T_a as a function of depth
#starting from known T_a at D
def get_T_a(z):
    integral, error = quad(get_d_T_a_dz, D, z)
    if Jenkins_ambient:
        T_a = -1.9 + integral
    else:
        T_a = -2.05 + integral
    #T_a = 0
    return T_a

#integrates derivative to find S_a as a function of depth
#starting from known S_a at D
def get_S_a(z):
    integral, error = quad(get_d_S_a_dz, D, z)
    if Jenkins_ambient:
        S_a = 34.5 + integral
    else:
        S_a = 34.65
    return S_a

#calculates rho_a as a function of T_a and S_a (which depend on depth)
def get_rho_a(z):
    S_a = get_S_a(z)
    T_a = get_T_a(z)
    return 1000 + rho_l * (beta_s * S_a + beta_T * (4 - T_a))

#specifies rate of change of T_a with depth
def get_d_T_a_dz(z):
    if Jenkins_ambient:
        return (-2.18 + 1.9) / 1115
    else:
        return (-1.85 + 2.05) / 1115

#specifies rate of change of S_a with depth
def get_d_S_a_dz(z):
    if Jenkins_ambient:
        return (34.71 - 34.5) / 1115
    else:
        return (34.35 - 34.65) / 1115

#calculates rate of change of rho_a with depth
def get_d_rho_a_dz(z):
    return rho_l * (beta_s * get_d_S_a_dz(z) - beta_T * get_d_T_a_dz(z))

#calculates depth at center of plume as a function of distance along shelf
def get_z(H, s):
    return D + s * sin_theta - H / 2

#numerator of coefficient of X in dY/ds
def get_A(z):
    S_a = get_S_a(z)
    return 1 / rho_l * get_d_rho_a_dz(z) - beta_s * S_a * c_l / L * lamda * sin_theta

#numerator of coefficient of Y^1/3 in dY/ds
def get_B(z):
    S_a = get_S_a(z)
    T_a = get_T_a(z)
    return beta_s * S_a * c_l / L * (tau * S_a + T_a - lamda * z)

#denominator of coefficients in dY/ds
def get_C(z):
    E = E_0 * sin_theta
    return C_d / E ** 3 / g / sin_theta

#calculates plume thickness
def get_H(y):
    E = E_0 * sin_theta
    X = get_X(y)
    Y = get_Y(y)
    return E * X / Y ** (1/3)

#calculates plume velocity
def get_U(y):
    E = E_0 * sin_theta
    Y = get_Y(y)
    return Y ** (1/3) / E

#calculates density deficit
def get_del_rho(y):
    U = get_U(y)
    H = get_H(y)
    return rho_l * C_d / g / sin_theta * U ** 2 / H

#calculates liquidus temperature
def get_T_L(S, z):
    return T_m + lamda * z - tau * (S - S_s)

def get_del_T_a(z):
    T_a = get_T_a(z)
    S_a = get_S_a(z)
    return T_a - get_T_L(S_a, z)

#calculates precipitation
def get_p(y, z):
    E = E_0 * sin_theta
    U = get_U(y)
    del_T_a = get_del_T_a(z)
    X = get_X(y)
    return c_l / L * (E * U * del_T_a - lamda * X * sin_theta)

#HU = X = y[0]
def get_X(y):
    return y[0]

#(dX/ds) ^ 3 = Y = y[1]
def get_Y(y):
    return y[1]

def derivative(y, s):
    X = get_X(y) #HU
    Y = get_Y(y) #(dX/ds) ^ 3
    z = get_z(get_H(y), s)
    dX = Y ** (1/3)
    dY = get_A(z) / get_C(z) * X + get_B(z) / get_C(z) * Y ** (1/3)
    return [dX, dY]

def basic_arrays_from_y(s, y):
    H_diff = []
    U_diff = []
    del_rho_diff = []
    p_diff = []
    for (vect, s0) in zip(y, s):
        z = get_z(get_H(y), s0)
        H_diff.append(get_H(vect))
        U_diff.append(get_U(vect))
        del_rho_diff.append(get_del_rho(vect))
        #p_diff.append(get_p(vect, z))
    return H_diff, U_diff, del_rho_diff, p_diff

def basic_plot(s, data, title):
    plt.plot(s, data)
    plt.title(title)
    plt.show()

s = np.linspace(405e3, 600e3)
y0 = [2.988119957536042, (9.915198356958679e-6) ** 3]
y = odeint(derivative, y0, s)

H_diff, U_diff, del_rho_diff, p_diff = basic_arrays_from_y(s, y)

print([H_diff[0], U_diff[0], del_rho_diff[0]])

# all_arrays = [H_diff, U_diff, del_rho_diff, p_diff]
# all_titles = ["H", "U", "delta rho", "p"]
all_arrays = [H_diff, U_diff, del_rho_diff]
all_titles = ["H", "U", "delta ro", "p"]

for data, title in zip(all_arrays, all_titles):
    basic_plot(s, data, title)