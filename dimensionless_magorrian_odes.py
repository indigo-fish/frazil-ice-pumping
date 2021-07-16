# -*- coding: utf-8 -*-
"""
alternate attempt at coding Magorrian Wells equations
using simplified dimensionless system
"""
import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#defines the one parameter still needed in equations
C_d = 2.5e-3 #drag coefficient
E_0 = 3.6e-2 #max entrainment coefficient
theta = 0.1
sin_theta = math.sin(theta)
gamma = C_d / E_0 / sin_theta #single necessary parameter

def get_u(y):
    return y[1] / y[0]

def get_b(y):
    return y[0] ** 2 / y[1]

def get_chi(y):
    return y[2] / y[0]

def get_theta(y):
    return y[3] / y[0]

def dy0_ds(y):
    u = get_u(y)
    return 3 / 2 * math.abs(u)

def dy1_ds(y):
    u = get_u(y)
    return (2 + gamma) * y[2] / u - gamma * u * math.abs(u)

def dy2_ds(y):
    u = get_u(y)
    b = get_b(y)
    return 3 / 2 * math.abs(u) - b * u

def dy3_ds(y):
    u = get_u(y)
    return 3 / 2 * math.abs(u)

def derivative(y, t):
    dy0 = dy0_ds(y)
    dy1 = dy1_ds(y)
    dy2 = dy2_ds(y)
    dy3 = dy3_ds(y)
    return [dy0, dy1, dy2, dy3]