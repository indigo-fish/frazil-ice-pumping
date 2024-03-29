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
lamda = 7.61e-4 #depth dependence of freezing point
L = 3.35e5 #latent heat of fusion for ice
c_s = 2.009e3 #specific heat capacity for ice
c_l = 3.974e3 #specific heat capacity for seawater
beta_s = 7.86e-4 #haline contraction coefficient
beta_T = 3.87e-5 #thermal expansion coefficient
g = 9.81 #gravitational acceleration
S_s = 0 #salt concentration in ice
#I don't know rho_a, rho_l, rho_s
rho_s = 916.8
nu = 1.95e-6
#T_a = 1 #ambient water temperature - specified in get_T_a(z) instead
D = -1400 #start depth
D_at_600 = -285 #end depth
Ta_at_0 = -1.9 #start ambient temperature
Ta_at_600 = -2.18 #end ambient temperature
Sa_at_0 = 34.5 #start ambient salinity
Sa_at_600 = 34.71 #end ambient salinity
#theta = 0.1
#sin_theta = (D_at_600 - D)/600e3 #slope of ice shelf
U_T = 0 #tide velocity
get_R = 0 #global variable for radius set by later code
k_l = .569 #thermal conductivity
T_deficit = 0.02 #supercooling
t_ice = .05e-3 #constant ice crystal thickness

my_precipitation = True #chooses between Stokes and Jenkins drag
Jenkins_ambient = True #chooses between ideal and Amery ice shelf conditions
vary_radius = True #chooses between fixed crystal radius and growing with plume
simplified_plume = False #chooses whether or not to set small terms to zero after frazil formation
linear_ice = True

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
        T_a = Ta_at_0 + integral
    else:
        T_a = -2.05 + integral
    #T_a = 0
    return T_a

#integrates derivative to find S_a as a function of depth
#starting from known S_a at D
def get_S_a(z):
    integral, error = quad(get_d_S_a_dz, D, z)
    if Jenkins_ambient:
        S_a = Sa_at_0 + integral
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
        return (Ta_at_600 - Ta_at_0) / (D_at_600 - D)
    else:
        return (-1.85 + 2.05) / (D_at_600 - D)

#specifies rate of change of S_a with depth
def get_d_S_a_dz(z):
    if Jenkins_ambient:
        return (Sa_at_600 - Sa_at_0) / (D_at_600 - D)
    else:
        return (34.35 - 34.65) / (D_at_600 - D)

#calculates rate of change of rho_a with depth
def get_d_rho_a_dz(z):
    return rho_l * (beta_s * get_d_S_a_dz(z) - beta_T * get_d_T_a_dz(z))

"""
functions which produce various other (generally non-constant)
values used in the system of differential equations
"""
#calculates temperature from H*U*delta_T and H*U
def get_T(y, z):
    result = T_m + lamda * z - tau * (get_S(y, z) - S_s) + y[3] / y[0]
    return result

#calculates liquidus temperature
def get_T_L(S, z):
    return T_m + lamda * z - tau * (S - S_s)

#calculates salinity from H*U*delta_T, H*U*delta_rho/rho_l and H*U
def get_S(y, z):
    S_a = get_S_a(z)
    temp = beta_s * S_a + beta_T * (T_m + lamda * z + tau * S_s + y[3] / y[0]) - y[2] / y[0]
    result = temp / (beta_s + beta_T * tau)
    return result

#quadratic coefficient a used to calculate M
def get_a(z):
    T_L_S_s = get_T_L(S_s, z)
    return 1 - c_s / L * (T_s - T_L_S_s)

#quadratic coefficient b used to calculate M
def get_b(T, S, z):
    T_L_S = get_T_L(S, z)
    T_L_S_s = get_T_L(S_s, z)
    return St_m * (1 - c_s / L * (T_s - T_L_S) - St * c_l / L * (T - T_L_S_s))

#quadratic coefficient c used to calculate M
def get_c(T, S, z):
    T_L_S = get_T_L(S, z)
    return - St_m * St * c_l / L * (T - T_L_S)

#calculates dimensionless coefficient for melt rate using quadratic formula
def get_M(T, S, z):
    a = get_a(z)
    b = get_b(T, S, z)
    c = get_c(T, S, z)
    result = (- b + math.sqrt((b ** 2 - 4 * a * c))) / (2 * a)
    return result

#calculates velocity using (H*U^2) / (H*U)
def get_U(y):
    return y[1] / y[0]

#calculates plume thickness using (H*U)^2 / (H*U)
def get_H(y):
    return y[0] ** 2 / y[1]

#calculates density deficit using H*U*delta_rho/rho_l and H*U
def get_del_rho(y):
    return rho_l * y[2] / y[0]

#calculates interfacial temperature by solving system of equations
def get_T_i(y, z):
    S_a = get_S_a(z)
    T_i, S_i = fsolve(interfacial_system, [get_T_L(S_a, z), S_a], args = (get_M(get_T(y, z), get_S(y, z), z), get_U(y), get_T(y, z), get_S(y, z), z))
    return T_i

#calculates interfacial salinity by solving system of equations
def get_S_i(y, z):
    S_a = get_S_a(z)
    T_i, S_i = fsolve(interfacial_system, [get_T_L(S_a, z), S_a], args = (get_M(get_T(y, z), get_S(y, z), z), get_U(y), get_T(y, z), get_S(y, z), z))
    return S_i

#calculates effective temperature due to latent heat released by melting
def get_T_eff(T_i):
    return T_i - L / c_l + c_s / c_l * (T_s - T_i)

#calculates effective density due to latent heat released by melting
def get_rho_eff(y, z):
    S_a = get_S_a(z)
    T_a = get_T_a(z)
    return rho_l * (beta_s * (S_a - S_s) - beta_T * (T_a - get_T_eff(get_T_i(y, z))))

#calculates difference between T_a and freezing temp at salinity S_a
def get_del_T_a(z):
    T_a = get_T_a(z)
    S_a = get_S_a(z)
    return T_a - get_T_L(S_a, z)

#calculates precipitation rate assuming Stokes drag/buoyancy balance and spherical ice crystals
def get_p(y, z):
    if my_precipitation:
        phi = get_phi(y, z)
        if vary_radius:
            if phi > 0:
                R = get_R_direct_stokes(get_H(y))
            else:
                R = 5e-3
        else:
            R = get_R
        v_rel = stokes_velocity(R) #vertical ice velocity relative to surrounding liquid
        result = - rho_s / rho_l * phi * v_rel
    else:
        phi = get_phi(y, z)
        if vary_radius:
            if phi > 0:
                R = get_R_direct_stokes(get_H(y))
                # #finds R for jenkins velocity but is not currently being used
                # v = stokes_velocity(R)
                # v, R = fsolve(solve_v_R_system, [v, R], args=(get_H(y)))
            else:
                R = 5e-3
        else:
            R = get_R
        epsilon = 0.02
        Re = get_reynolds(stokes_velocity(R), R)
        local_drag = get_local_drag(Re)
        W_d = math.sqrt(2 * (rho_l - rho_s) / rho_l * g * 2 * epsilon * R / local_drag)
        # #uses critical velocity to determine if ice crystals able to precipitate
        # #not currently working
        # U = get_U(y)
        # r_e = (3 / 2 * epsilon) ** (1 / 3) * R
        # U_c = .05 * (rho_l - rho_s) / rho_l * g * 2 * r_e / C_d
        # if U < U_c:
        #     result = - rho_s / rho_l * phi * W_d * math.sqrt(1 - sin_theta ** 2) * (1 - U ** 2 / U_c ** 2)
        # else:
        #     result = 0
        H = get_H(y)
        sin_theta = get_sin_theta(z, H)
        result = - rho_s / rho_l * phi * W_d * math.sqrt(1 - sin_theta ** 2)
    return result

#calculates max velocity according to Stokes drag formula
def stokes_velocity(R):
    return 2 / 9 * R ** 2 * g * (rho_l - rho_s) / (nu * rho_l)

#calculates max velocity according to drag formula in Jenkins and Bombosch
def jenkins_velocity(R, t_ice):
    epsilon = t_ice/R
    W_d = stokes_velocity(R) / 3 / epsilon
    count = 0
    while count < 5:
        Re = get_reynolds(W_d, R)
        local_drag = get_local_drag(Re)
        W_d = math.sqrt(2 * (rho_l - rho_s) / rho_l * g * 2 * epsilon * R / local_drag)
        count += 1
    return W_d

#uses Stokes drag formula to predict max crystal radius given plume thickness
def get_R_direct_stokes(H):
    cube = 9 / 2 * k_l * T_deficit * H * nu * rho_l / (L * g * t_ice * (rho_l - rho_s) * rho_s)
    return cube ** (1/3)

#calculates local drag coefficient used in Jenkins precipitation velocity
def get_local_drag(Re):
    log_Re = math.log10(Re)
    log_drag = 1.386 - .892 * log_Re + .111 * log_Re ** 2
    return 10 ** log_drag

#calculates local Reynolds number / kinematic viscosity
def get_reynolds(v, R):
    result = abs(2 * v * R / nu)
    return result

#calculates max crystal radius given plume thickness and velocity
def get_R_mean(H, v_sn):
    return k_l * T_deficit * H / rho_s / L / t_ice / v_sn

#systen of equations for radius and Jenkins drag velocity
def solve_v_R_system(vect, H):
    v, R = vect
    func1 = jenkins_velocity(R, t_ice) - v
    func2 = get_R_mean(H, v) - R
    return [func1, func2]

#returns velocity of ice sheet at given distance from grounding line
#for calculating thickness of marine ice
def get_sheet_v(s):
    if s < 100:
        m_per_y = 800 - 2 * s
    elif s < 200:
        m_per_y = 600 - 3 * (s - 100)
    elif s < 300:
        m_per_y = 300
    elif s < 400:
        m_per_y = 300 + 2 * (s - 300)
    elif s < 500:
        m_per_y = 500 + 5 * (s - 400)
    else:
        m_per_y = 1000
    result = m_per_y / 365 / 24 / 3600
    return result

#calculates phi by dividing U*phi / U
def get_phi(y, z):
    return y[4] / get_U(y)

#describes varying sin theta to allow calculation of non-linear z
def get_sin_theta(z, H):
    if linear_ice:
        # s = fsolve(system_get_s, z / (D_at_600) * 600e3, args=(z))
        # sin_theta = (get_z_ice(s + 5) - get_z_ice(s - 5)) / 10
        sin_theta = (D_at_600 - D)/600e3
    else:
        s = fsolve(system_get_s, z / (D_at_600 - D) * 600e3, args=(z))
        #need to figure out how to find sin_theta!
        sin_theta = (D_at_600 - D)/600e3
    return sin_theta

def get_z_ice(s):
    if linear_ice:
        return D + s * (D_at_600 - D)/600e3
    else:
        depth = D
        if s < 0:
            depth += 0
        elif s < 20e3:
            depth += s * (-1800 + 2500) / 20e3
        elif s < 120e3:
            depth += (-1800 + 2500)
            depth += (s - 20e3) * (-1000 + 1800) / 100e3
        elif s < 210e3:
            depth += (-1000 + 2500)
            depth += (s - 120e3) * (-700 + 1000) / 90e3
        elif s < 340e3:
            depth += (-700 + 2500)
            depth += (s - 210e3) * (-600 + 700) / 130e3
        elif s < 580e3:
            depth += (-600 + 2500)
            depth += (s - 340e3) * (-200 + 600) / 240e3
        else:
            depth += 2500
        return depth
        

#calculates depth at center of plume as a function of distance along shelf
def get_z(H, s):
    return get_z_ice(s) - H / 2

def system_get_s(s, z):
    return get_z_ice(s) - z

#system of equations which is solved to find T_i and S_i, given M, U, T and S
def interfacial_system(vect, M, U, T, S, z):
    T_i, S_i = vect
    func1 = rho_l * L * M * abs(U) + rho_s * c_s * (T_i - T_s) * M * abs(U) - rho_l * c_l * St * abs(U) * (T - T_i)
    func2 = get_T_L(S_i, z) - T_i
    return [func1, func2]

"""
defines each of the linear differential equations
in terms of the variables which make it easier to solve
"""
#derivative of H*U
def dy0_ds(y, z):
    U = get_U(y)
    H = get_H(y)
    sin_theta = get_sin_theta(z, H)
    e = E_0 * sin_theta * abs(U)
    m = get_M(get_T(y, z), get_S(y, z), z) * abs(U)
    p = get_p(y, z)
    if simplified_plume:
        p = 0
        if z > get_z(20, 400e3):
            m = 0
    return [e + m + p, e, m, p]

#derivative of H*U^2
def dy1_ds(y, z):
    H = get_H(y)
    U = get_U(y)
    phi = get_phi(y, z)
    delta_rho = get_del_rho(y)
    sin_theta = get_sin_theta(z, H)
    liquid_buoyancy = g * sin_theta * H * delta_rho / rho_l
    ice_buoyancy = g * sin_theta * H * phi * (1 - rho_s / rho_l)
    drag = - C_d * U * math.sqrt(U ** 2 + U_T ** 2)
    if simplified_plume:
        ice_buoyancy = 0
    result = liquid_buoyancy + ice_buoyancy + drag
    return [result, liquid_buoyancy, ice_buoyancy, drag]

#derivative of H*U*delta_rho / rho_l
def dy2_ds(y, z):
    U = get_U(y)
    del_rho_eff = get_rho_eff(y, z)
    m = get_M(get_T(y, z), get_S(y, z), z) * abs(U)
    H = get_H(y)
    sin_theta = get_sin_theta(z, H)
    drho_a_ds = get_d_rho_a_dz(z) * sin_theta
    S_a = get_S_a(z)
    T_a = get_T_a(z)
    T_i = get_T_i(y, z)
    p = get_p(y, z)
    ambient_gradient = drho_a_ds / rho_l * y[0]
    melt_density = del_rho_eff / rho_l * m
    melt_density = beta_s * S_a * m
    T_eff = get_T_eff(T_i)
    precip_heat = (beta_s * S_a - beta_T * (T_a + L / c_l - c_s / c_l * T_i)) * p - beta_T * (T_a - T_eff) * m
    get_dy4 = dy4_ds(y, z)
    phi_heat = beta_T * rho_s / rho_l * L / c_l * get_dy4[0] * H
    if simplified_plume:
        phi_heat = 0
        precip_heat = beta_s * S_a * p
        if z > get_z(20, 400e3):
            melt_density = 0
    result = ambient_gradient + melt_density + precip_heat + phi_heat
    return [result, ambient_gradient, melt_density, precip_heat, phi_heat]

#derivative of H*U*delta_T
def dy3_ds(y, z):
    U = get_U(y)
    H = get_H(y)
    sin_theta = get_sin_theta(z, H)
    e = E_0 * sin_theta * abs(U)
    del_T_a = get_del_T_a(z)
    m = get_M(get_T(y, z), get_S(y, z), z) * abs(U)
    T_i = get_T_i(y, z)
    T_eff = get_T_eff(T_i)
    T_L_S_s = get_T_L(S_s, z)
    p = get_p(y, z)
    e_temp = del_T_a * e
    m_temp = m * (T_eff - T_L_S_s)
    depth_cooling = - lamda * sin_theta * y[0]
    p_interfacial = c_s / c_l * T_i * p
    p_latent = - (L / c_l + T_m + lamda * z) * p
    get_dy4 = dy4_ds(y, z)
    H = get_H(y)
    phi_latent = rho_s / rho_l * L / c_l * get_dy4[0] * H
    if simplified_plume:
        p_interfacial = 0
        phi_latent = 0
        if z > get_z(20, 400e3):
            m_temp = 0
    result = e_temp + m_temp + depth_cooling + p_interfacial + p_latent + phi_latent
    return [result, e_temp, m_temp, depth_cooling, p_interfacial, p_latent, phi_latent]

#derivative of U*phi
def dy4_ds(y, z):
    T = get_T(y, z)
    S = get_S(y, z)
    T_L = get_T_L(S, z)
    if T > T_L:
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        T_i = get_T_i(y, z)
        S_i = get_S_i(y, z)
        U = get_U(y)
        p = get_p(y, z)
        H = get_H(y)
        sin_theta = get_sin_theta(z, H)
        e = E_0 * sin_theta * abs(U)
        m = get_M(T, S, z) * abs(U)
        T_a = get_T_a(z)
        S_a = get_S_a(z)
        get_dy0 = dy0_ds(y, z)
        H = get_H(y)
        transport_freezing = (T_m + lamda * z) * get_dy0[0] * rho_l * c_l / L / rho_s / H
        salt_ambient = - tau * (e * S_a) * rho_l * c_l / L / rho_s / H
        salt_interfacial = - tau * (m * S_i - St_m * U * (S - S_i)) * rho_l * c_l / L / rho_s / H
        depth_freezing = lamda * y[0] * sin_theta * rho_l * c_l / L / rho_s / H
        ambient = - e * T_a * rho_l * c_l / L / rho_s / H
        interfacial = - m * T_i * rho_l * c_l / L / rho_s / H
        p_latent = - c_s / c_l * p * T_i * rho_l * c_l / L / rho_s / H
        heat_transp = St * U * (T - T_i) * rho_l * c_l / L / rho_s / H
        p_vol = rho_l * p / rho_s / H
        if simplified_plume:
            p_latent = 0
            heat_transp = 0
            transport_freezing = 0
            if z > get_z(20, 400e3):
                salt_interfacial = 0
                interfacial = 0
        res_sum = transport_freezing + salt_ambient + salt_interfacial + depth_freezing + ambient + interfacial + p_latent + heat_transp + p_vol
        result = [res_sum, salt_interfacial, salt_ambient + ambient, depth_freezing + p_vol + transport_freezing, 0, interfacial, p_latent, heat_transp, 0]
    return result

"""
re-expresses system of differential equations as a vector
"""
def derivative(y, s):
    z = get_z(get_H(y), s)
    dy0 = dy0_ds(y, z)
    dy1 = dy1_ds(y, z)
    dy2 = dy2_ds(y, z)
    dy3 = dy3_ds(y, z)
    dy4 = dy4_ds(y, z)
    if 405e3 < s < 406.2e3: print(str(y[0]) + " " + str(dy0[0]))
    return [dy0[0], dy1[0], dy2[0], dy3[0], dy4[0]]

"""
comes up with initial values using analytical solutions
"""

"""
system of equations which is solved to find M, T, S
(under assumption that M is small, so do not need to calculate T_i, S_i
for calculation of initial y values)
"""
#simplified version with assumption of small M
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

#exact version used on later iterations
def repeat_init_solve(vect, E, X, U, T_i, S_i):
    z = get_z(H0, X)
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
various functions for post-processing results of differential equations
"""
#given solutions of differential equations
#creates arrays of more meaningful quantities H, U, del_T, del_rho, phi
def basic_arrays_from_y(s, y):
    H_diff = []
    U_diff = []
    del_T_diff = []
    del_rho_diff = []
    phi_diff = []
    for (vect, s0) in zip(y, s):
        z = get_z(get_H(vect), s0)
        H = get_H(vect)
        U = get_U(vect)
        T = get_T(vect, z)
        S = get_S(vect, z)
        delta_rho = get_del_rho(vect)
        phi = get_phi(vect, z)
        H_diff.append(H)
        U_diff.append(U)
        del_T_diff.append(T - get_T_L(S, z))
        del_rho_diff.append(delta_rho)
        phi_diff.append(phi)
    return H_diff, U_diff, del_T_diff, del_rho_diff, phi_diff

#given solutions of differential equations
#creates arrays of terms in derivatives
def derivative_arrays_from_y(s, y):
    dy0 = []
    dy1 = []
    dy2 = []
    dy3 = []
    dy4 = []
    for (vect, s0) in zip(y, s):
        z = get_z(get_H(vect), s0)
        dy0.append(dy0_ds(vect, z))
        dy1.append(dy1_ds(vect, z))
        dy2.append(dy2_ds(vect, z))
        dy3.append(dy3_ds(vect, z))
        dy4.append(dy4_ds(vect, z))
    return dy0, dy1, dy2, dy3, dy4

#given solutions of differential equations
#creates arrays of precipitation and marine ice accumulation
def precipitation_arrays_from_y(s, y):
    p_levels = []
    ice_depth = 0
    ice_depths = []
    for index, s0 in enumerate(s):
        vect = y[index]
        z = get_z(get_H(vect), s0)
        p = abs(get_p(vect, z))
        p_levels.append(p)
        if index == 0:
            ice_depths.append(0)
        else:
            ice_depth += p * (s0 - s[index - 1]) / get_sheet_v(s0)
            ice_depths.append(ice_depth)
    return p_levels, ice_depths

#given solutions of differential equations
#creates array of ice crystal radius with Stokes drag
def r_array_from_y(s, y):
    r_diff = []
    for (vect, s0) in zip(y, s):
        z = get_z(get_H(vect), s0)
        if get_phi(vect, z):
            r = 1000 * get_R_direct_stokes(get_H(vect))
        else:
            r = 0
        r_diff.append(r)
    return r_diff

#given solutions of differential equations
#creates arrays of HU and HU^2
def get_HU_array(s, y):
    HU_diff = []
    HU2_diff = []
    for (vect, s0) in zip(y, s):
        HU_diff.append(vect[0])
        HU2_diff.append(vect[1])
    return HU_diff, HU2_diff

#given array of vectors with elements corresponding to different radii
#plots elements as a function of distance
def plot_multi_radius(s, data, radii, title, ylabel):
    for data_line, r_label in zip(data, radii):
        plt.plot(s / 1000, data_line, label=(str(r_label * 1000) + " mm"))
    plt.title(title)
    plt.xlabel("Distance Along Ice Shelf (km)")
    plt.ylabel(ylabel)
    #plt.legend()
    plt.show()

#given array of vectors with elements corresponding to terms in derivatives
#plots elements as a function of distance
def plot_derivative(s, data, radius, title, labels):
    plt.plot(s, data)
    plt.legend(labels)
    #title += " at radius " + str(radius)
    plt.title(title)
    plt.legend(labels)
    plt.show()

#helper function for plot_phi_radii
#which creates arrays of indexed location at different r values
def get_index_data(data, indices, vert_place = 0):
    indexed_data = []
    for data_line in data:
        vect = []
        for index in indices:
            point = data_line[index]
            if np.isscalar(point):
                vect.append(point)
            else:
                vect.append(point[vert_place])
        indexed_data.append(vect)
    return indexed_data

#helper function for plot_phi_radii
#which creates arrays of ln radius and ln phi
def get_log_log(x_set, data):
    log_x = []
    log_data = []
    for x, vect in zip(x_set, data):
        log_vect = []
        for elem in vect:
            log_vect.append(math.log(abs(elem)))
        log_x.append(math.log(x))
        log_data.append(log_vect)
    return log_x, log_data

#given array of all phi values for different radii
#plots phi as a function of radius for a specified index
#also used for plotting other values as a function of radius in spite of name
def plot_phi_radii(radii, data, precip_state, indices, title, vert_place = 0):
    #extracts data points for phi at specified indices
    phis = get_index_data(data, indices, vert_place)
    #converts r and phi data into log-log form
    ln_r, ln_phi = get_log_log(radii, phis)
    #plots ordinary phi-r relationship
    plt.plot(radii, phis, marker = '.', linewidth = 0)
    if precip_state:
        title += "Stokes Drag"
    else:
        title += "Jenkins Drag"
    plt.title(title)
    plt.legend(indices)
    plt.show()
    #plots log-log phi-r relationship
    plt.plot(ln_r, ln_phi, marker = '.', linewidth = 0)
    index = 0
    while index < len(indices):
        #constructs lines of best fit for phi-r data from each index
        ln_phi_fit = []
        for vect in ln_phi:
            ln_phi_fit.append(vect[index])
        m, b = np.polyfit(ln_r, ln_phi_fit, 1)
        b0 = math.exp(b)
        equation = "phi = " + str(round(b0, 9)) + " * r ^ " + str(round(m, 3))
        x = np.array(ln_r)
        plt.plot(x, m * x + b)
        plt.annotate(equation, (.1, .1 + .05 * index), xycoords = 'figure fraction')
        index += 1
    plt.legend(indices)
    plt.title(title)
    plt.show()

def plot_dy4s(s, data, radii, precip_state, title):
    dy4_total = []
    for radius, row in zip(radii, data):
        dy4_t_row = []
        for vect in row:
           dy4_t_row.append(vect[0])
        dy4_total.append(dy4_t_row)
    
    for data_line, r_label in zip(dy4_total, radii):
        plt.plot(s, data_line, label=str(r_label))
    plt.title(title)
    plt.legend()
    plt.show()

#plots beginning at 400km for direct comparison with simplified plume
def plot_while_frazil(s, data, title):
    s_cropped = []
    data_cropped = []
    for s0, data0 in zip(s, data):
        if s0 > 400e3:
            s_cropped.append(s0)
            data_cropped.append(data0)
    plt.plot(s_cropped, data_cropped)
    plt.title(title)
    plt.show()

def plot_radii(s, data, title, ylabel):
    s_crop = []
    r_crop = []
    for s0, data0 in zip(s, data):
        if data0 > 0:
            s_crop.append(s0 / 1000)
            r_crop.append(data0)
    plt.plot(s_crop, r_crop)
    plt.title(title)
    plt.ylabel("Crystal Radius (mm)")
    plt.title(title)
    plt.xlabel("Distance Along Ice Shelf (km)")
    plt.show()

#sets initial distance along slope at which analytical solutions are used
X0 = 0.01
z0 = get_z(0, 0.01)
sin_theta0 = get_sin_theta(z0, 0)
E0 = E_0 * sin_theta0


#solves (simplified) analytic system of equations for M, T, S
S_a0 = get_S_a(z0)
T_a0 = get_T_a(z0)
M0, T0, S0 = fsolve(solve_init_system, [E0 * St / (E0 + St) * c_l * get_del_T_a(z0) / L, T_a0, S_a0], args = (E0, z0))

"""
converts M, T, S values into values for other terms
allowing use of full versions of equations 29, 30 in Magorrian Wells
"""
H0 = 2 / 3 * (E0 + M0) * X0
del_T0 = T0 - get_T_L(S0, z0)
del_rho0 = - rho_l * (beta_s * (S0 - S_a0) - beta_T * (T0 - T_a0))
U0 = math.sqrt(2 * (E0 + M0 / (3 * C_d + 4 * (E0 + M0)))) * math.sqrt(del_rho0 / rho_l * g * sin_theta0 * X0)
T_i0, S_i0 = fsolve(interfacial_system, [get_T_L(S_a0, z0), S_a0], args = (M0, U0, T0, S0, z0))

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
    U0 = math.sqrt((2 * ((E0 + M0) / (3 * C_d + 4 * (E0 + M0))) * del_rho0 / rho_l * g * sin_theta0 * X0))
    T_i0, S_i0 = fsolve(interfacial_system, [get_T_L(S_a0, z0), S_a0], args = (M0, U0, T0, S0, z0))
    i_vals.append(i)
    del_T_vals.append(del_T0)
    del_rho_vals.append(del_rho0)
    U_vals.append(U0)
    M_vals.append(M0)
    i += 1

#converts values into format most useful for differential equations
y0_0 = H0 * U0
y0_1 = H0 * U0 ** 2
y0_2 = H0 * U0 * del_rho0 / rho_l
y0_3 = H0 * U0 * del_T0
y0_4 = 0

#puts these initial values in a vector for solving equations
y0 = [y0_0, y0_1, y0_2, y0_3, y0_4]

#y0 = [9.20816705e-06,  8.04886127e-08,  9.61161561e-08, -5.18938969e-05, 3.59298686e-06]
print(y0)

#defines distances along slope to record results
list1 = np.linspace(0, 300e3, 25)
list2 = np.linspace(300e3, 600e3, 150)
s = np.concatenate((list1, list2))

precip_type_count = 0
my_precipitation = True
fig_num = 0

# y = odeint(derivative, y0, s)

# H_diff, U_diff, del_T_diff, del_rho_diff, phi_diff = basic_arrays_from_y(s, y)
# dy0, dy1, dy2, dy3, dy4 = derivative_arrays_from_y(s, y)
# p_levels, ice_depths = precipitation_arrays_from_y(s, y)
# r_diff = r_array_from_y(s, y)
# all_arrays = [H_diff, U_diff, del_T_diff, del_rho_diff, phi_diff, p_levels, ice_depths, r_diff]
# all_titles = ["H", "U", "delta T", "delta rho", "phi", "precipitation", "accumulation", "radius"]

# for data, title in zip(all_arrays, all_titles):
#     plt.plot(s, data)
#     plt.title(title)
#     plt.show()

while precip_type_count < 1:
#while precip_type_count < 1:
    #radii = [.01e-3, .05e-3, .1e-3, .5e-3, 1e-3, 5e-3]
    #radii = [.01e-3, .03e-3, .05e-3, .07e-3, .09e-3, .1e-3, .3e-3, .5e-3, .7e-3, .9e-3, 1e-3, 3e-3, 5e-3]
    #radii = np.linspace(.01e-3, 5e-3)
    radii = [.1e-3]
    #radii = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    
    all_H = []
    all_U = []
    all_del_T = []
    all_del_rho = []
    all_phi = []
    all_p = []
    all_ice_depths = []
    all_dy4 = []
    all_dy1 = []
    all_HU = []
    all_HU2 = []
    all_r = []
    titles = ["Plume Thickness", "Plume Velocity", "Temperature Deficit", "Density Deficit", "Solid Fraction", "Precipitation", "Accumulation", "Stokes Drag Crystal Radius"]
    ylabels = ["Plume Thickness (m)", "Plume Velocity (m/s)", "Temperature Deficit (K)", "Density Deficit ($kg/m^3$)", "Solid Fraction", "Precipitation (m/s)", "Accumulation (m)", "Crystal Radius (m)"]
    
    count = 0
    while (count < len(radii)):
        get_R = radii[count]
        #defines distances along slope to record results
        y = odeint(derivative, y0, s)
        print(y[len(y) - 1])
        
        H_diff, U_diff, del_T_diff, del_rho_diff, phi_diff = basic_arrays_from_y(s, y)
        dy0, dy1, dy2, dy3, dy4 = derivative_arrays_from_y(s, y)
        p_levels, ice_depths = precipitation_arrays_from_y(s, y)
        HU_diff, HU2_diff = get_HU_array(s, y)
        r_diff = r_array_from_y(s, y)
        
        z_diff = []
        for s0 in s:
            z_diff.append(get_z_ice(s0))
        
        plt.show()
        plt.plot(s, z_diff)
        plt.show()
        
        all_arrays = [H_diff, U_diff, del_rho_diff, p_levels]
        all_titles = ["H", "U", "delta rho", "precipitation"]
        
        # for data, title in zip(all_arrays, all_titles):
        #     plot_while_frazil(s, data, title)
        
        all_H.append(H_diff)
        all_U.append(U_diff)
        all_del_T.append(del_T_diff)
        all_del_rho.append(del_rho_diff)
        all_phi.append(phi_diff)
        all_p.append(p_levels)
        all_ice_depths.append(ice_depths)
        all_HU.append(HU_diff)
        all_HU2.append(HU2_diff)
        all_r.append(r_diff)
        set_of_dy = [dy0, dy1, dy2, dy3, dy4]
        all_dy4.append(dy4)
        all_dy1.append(dy1)
        
        dy0_labels = ["total", "entrainment", "melting", "precipitation"]
        dy1_labels = ["total", "liquid buoyancy", "ice buoyancy", "drag"]
        dy2_labels = ["total", "ambient gradient", "melt density", "precipitated heat", "frazil heat"]
        dy3_labels = ["total", "ambient temp", "melt temp", "depth cooling", "interfacial temp", "latent precipitation", "latent frazil"]
        dy4_labels = ["total", "transport freezing", "salt freezing", "depth freezing", "ambient melting", "interfacial frazil", "latent precipitation", "heat transport", "volume precipitation"]
        dy_titles = ["d(HU)/ds", "Contributions to Momentum Conservation", "d(HU del_rho/rho_l)/ds", "d(HU del_T)/ds", "d(U phi)/ds"]
        all_dy_labels = [dy0_labels, dy1_labels, dy2_labels, dy3_labels, dy4_labels]
        
        # #plots all derivatives for given radius
        # for data, title, labels in zip(set_of_dy, dy_titles, all_dy_labels):
        #     plot_derivative(s, data, get_R, title, labels)
        
        # #plots only derivative of U * phi
        # plot_derivative(s, dy4, get_R, "d(U phi)/ds", dy4_labels)
        
        # #plots only derivative of HU^2
        # if my_precipitation:
        #     title = "Stokes Drag: "
        # else:
        #     title = "Jenkins Drag: "
        # plot_derivative(s, dy1, get_R, title + "d(HU^2)/ds", dy1_labels)
        
        #iterates to next radius
        count += 1
    
    #plot_dy4s(s, all_dy4, radii, my_precipitation, "total dy4")
    
    all_arrays = [all_H, all_U, all_del_T, all_del_rho, all_phi, all_p, all_ice_depths, all_r]
    
    #plots each of H, U, del_rho, del_T, phi, p, accumulation
    #with variety of radii on same plot
    for array_set, var_title, ylabel in zip(all_arrays, titles, ylabels):
        var_title += " Along Ice Shelf"
        plot_multi_radius(s, array_set, radii, var_title, ylabel)
    
    plot_radii(s, r_diff, titles[len(titles) - 1], ylabels[len(ylabels) - 1])
    
    # #plots phi at specified indices as a function of radius
    # indices = [len(s) - 1, int(len(s) / 2), 3 * int(len(s) / 4)]
    # title = "phi as a function of radius: "
    # plot_phi_radii(radii, all_phi, my_precipitation, indices, title)
    
    # #plots dy4 at specified indices as a function of radius
    # title = "dy4 as a function of radius: "
    # plot_phi_radii(radii, all_dy4, my_precipitation, indices, title)
    
    # #plots dy1 at specified indices as a function of radius for comparison
    # title = "dy1 as a function of radius: "
    # plot_phi_radii(radii, all_dy1, my_precipitation, indices, title)
    
    #switches from Stokes Drag to Jenkins and Bombosch formula and repeats graphs
    my_precipitation = False
    precip_type_count += 1