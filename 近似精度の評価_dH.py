import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg, optimize
from scipy.integrate import solve_ivp

H_0 = 62.9
H_plusinf = 43.1
H_minusinf = 124
k_ref = 0.0019
E_a = 170.604
T_ref = 288.15
Rg = 0.008314

T_min, T_max = 278, 298
H_min, H_max = 42, H_0
Enz_min, Enz_max = 61, 80

n = 3
N = 100
c = 5
s = 5
num_piece = int(s*(s-1)/2)

T = np.random.uniform(T_min,T_max,(N,))
H = np.random.uniform(H_min,H_max,(N,))
Enz = np.random.uniform(Enz_min,Enz_max,(N,))

def k_rate(T):
    return k_ref*np.exp((E_a/Rg)*(1/T_ref-1/T))

def f(T,H,Enz):
    return -k_rate(T)*H*Enz

