# -*- coding: utf-8 -*-
"""
calculates most probable ice crystal radii
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
get_R = 0 #global variable for radius set by later code
my_precipitation = True #chooses between Stokes and Jenkins drag
Jenkins_ambient = True #chooses between ideal and Amery ice shelf conditions
k_l = .569 #thermal conductivity
epsilon = 0.02

#provides linear structure of density
rho_l = 1028

def get_R_mean(T_deficit, H, t_ice, v_sn):
    return k_l * T_deficit * H / rho_s / L / t_ice / v_sn

def stokes_velocity(R):
    return 2 / 9 * R ** 2 * g * (rho_l - rho_s) / (nu * rho_l)

def jenkins_velocity(R):
    W_d = stokes_velocity(R) / 3 / epsilon
    count = 0
    while count < 5:
        Re = get_reynolds(W_d, R)
        local_drag = get_local_drag(Re)
        square = 2 * (rho_l - rho_s) / rho_l * g * 2 * epsilon * R / local_drag
        W_d = math.sqrt(2 * (rho_l - rho_s) / rho_l * g * 2 * epsilon * R / local_drag)
        count += 1
    return W_d

#calculates local drag coefficient used in Jenkins precipitation velocity
def get_local_drag(Re):
    log_Re = math.log10(Re)
    log_drag = 1.386 - .892 * log_Re + .111 * log_Re ** 2
    return 10 ** log_drag

#calculates Reynolds number by dividing HU / kinematic viscosity
def get_reynolds(v, R):
    result = abs(2 * v * R / nu)
    return result

def solve_v_R_system(vect, T_deficit, H, t_ice):
    v, R = vect
    func1 = stokes_velocity(R) - v
    func2 = get_R_mean(T_deficit, H, t_ice, v) - R
    return [func1, func2]

H = 10
T_vals = [2e-4, 2e-3, 0.02]
labels = ["2e-4", "2e-3", "0.02", "0.2"]
t_ice = .05e-3
v_vals = np.linspace(1e-3, 1e-2, 200)
r_vals = []

for v in v_vals:
    r = []
    for T_deficit in T_vals:
        r.append(get_R_mean(T_deficit, H, t_ice, v))
    r_vals.append(r)

plt.plot(v_vals, r_vals, marker='.', linewidth=0)
plt.legend(labels)
plt.title("Crystal Radius as a Function of Rise Velocity")
plt.show()

v_vals = []
R_vals = []

H_vals = np.linspace(0.1, 50, 200)
for H in H_vals:
    v_vect = []
    R_vect = []
    for T_deficit in T_vals:
        v, R = fsolve(solve_v_R_system, [stokes_velocity(t_ice) / 2, t_ice], args = (T_deficit, H, t_ice))
        v_vect.append(v)
        R_vect.append(R)
    v_vals.append(v_vect)
    R_vals.append(R_vect)
plt.plot(H_vals, v_vals, marker='.', linewidth=0)
plt.legend(labels)
plt.title("Stokes Rise Velocity as a Function of Plume Width")
plt.show()
plt.plot(H_vals, r_vals, marker='.', linewidth=0)
plt.legend(labels)
plt.title("Stokes Crystal Radius as a Function of Plume Width")
plt.show()