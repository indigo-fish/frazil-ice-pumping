# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:32:12 2021

@author: ameli
"""

import sympy as sym
from math import sin

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
sin_theta = sin(theta)
S_a = 34 #ambient water salinity
#I don't know rho_a, rho_l, rho_s
rho_a = 1023.6
rho_l = 1023.6
rho_s = 916.8
#T_a is supposed to vary
T_a = 0

M = sym.Symbol('M')
T_i = sym.Symbol('T_i')
S_i = sym.Symbol('S_i')
T = sym.Symbol('T')
S = sym.Symbol('S')

def eq1(M, T_i, S_i, T, S):
    return M - c_l * St / (L + c_s * (T_i - T_s)) * (T - T_i)

def eq2(T_i, S_i, T, S):
    return c_l * St / (L + c_s * (T_i - T_s)) * (T - T_i) * (S_i - S_s) + St_m * (S - S_i)

def eq3(T_i, S_i, T, S):
    return T_m - tau * (S_i - S_s) - T_i

equations = (eq1(M, T_i, S_i, T, S), eq2(T_i, S_i, T, S), eq3(T_i, S_i, T, S))

result = sym.solve(equations, M, T_i, S_i, cubics = False)
print(result[0])
print(result[1])