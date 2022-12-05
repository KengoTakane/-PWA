import numpy as np
import statsmodels.api as sm
from scipy import linalg, optimize
from sklearn.svm import SVC

H_0 = 62.9
H_plusinf = 43.1
H_minusinf = 124
k_ref = 0.0019
E_a = 170.604
T_ref = 288.15
Rg = 0.008314

T_min, T_max = 278, 298
H_min, H_max = 40, 65
Enz_min, Enz_max = 90, 110


def k_rate(T):
    return k_ref*np.exp((E_a/Rg)*(1/T_ref-1/T))

def f_H(T,H,Enz):
    return -k_rate(T)*H*Enz

def f_Enz(T,H,Enz):
    return k_rate(T)*H*Enz


n = 3
N = 200
c = 5
s = 3

# row vector(1-dim array)
T = np.random.randint(T_min,T_max,(N,))
H = np.random.randint(H_min,H_max,(N,))
Enz = np.random.randint(Enz_min,Enz_max,(N,))