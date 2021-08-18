"""
exploration of when different drag conditions are more applicable
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
#get_R = 1e-3 #global variable for radius set by later code
my_precipitation = True #chooses between Stokes and Jenkins drag
Jenkins_ambient = True #chooses between ideal and Amery ice shelf conditions
epsilon = 0.02

#provides linear structure of density
rho_l = 1028

def stokes_drag(v, R):
    return 6 * math.pi * R * v * rho_l * nu

def stokes_velocity(R):
    return 2 / 3 * epsilon * R ** 2 * g * (rho_l - rho_s) / (nu * rho_l)

def jenkins_drag(v, R):
    Re = get_reynolds(v, R)
    local_drag = get_local_drag(Re)
    return rho_l * local_drag * (v ** 2) * (R ** 2)

def jenkins_velocity(R):
    W_d = stokes_velocity(R) / 3 / epsilon
    count = 0
    while count < 5:
        Re = get_reynolds(W_d, R)
        local_drag = get_local_drag(Re)
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

v_vals = np.linspace(1e-5, 0.01)
r_vals = [1e-6, 1e-5, 1e-4, 1e-2]

stokes = []
jenkins = []

for r in r_vals:
    stokes_v = []
    jenkins_v = []
    for v in v_vals:
        stokes_v.append(stokes_drag(v, r))
        jenkins_v.append(jenkins_drag(v, r))
    stokes.append(stokes_v)
    jenkins.append(jenkins_v)
    plt.plot(v_vals, stokes_v, label="Stokes drag " + str(r), marker=".")
    plt.plot(v_vals, jenkins_v, label="Jenkins drag " + str(r), marker="*")

plt.title("Drag as Function of Rise Velocity")
plt.legend()
plt.show()

r_vals = np.linspace(1e-6, 1e-3)

stokes = []
jenkins = []

for r in r_vals:
    stokes.append(stokes_velocity(r))
    jenkins.append(jenkins_velocity(r))

plt.plot(r_vals, stokes, label="Stokes drag")
plt.plot(r_vals, jenkins, label="Jenkins drag")
plt.legend()
plt.title("Max Velocity Due to Drag as a Function of Radius")
plt.show()