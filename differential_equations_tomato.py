import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode


H_0 = 62.9
H_plusinf = 43.1
H_minusinf = 124
k_ref = 0.0019
E_a = 170.604
T_ref = 288.15
Rg = 0.008314


def k(T):
    return k_ref*np.exp((E_a/Rg)*(1/T_ref-1/T))


def f(t, v): # H:v[0], Enz:v[1]
    return [-k*v[0]*v[1], k*v[0]*v[1]]