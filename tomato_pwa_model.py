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

# X : n×N dimension matrix, y : N dimension row vector(1-dim array)
X = np.array([T,H,Enz])
y_H = f_H(T,H,Enz)
y_Enz = f_Enz(T,H,Enz)

# Norm  :N×N dimension matrix, Norm_sort : N×N dimension matrix, C : N×c dimension matrix
Norm = np.empty((X.shape[1], X.shape[1]))
for i in range(X.shape[1]):
    for j in range(X.shape[1]):
        Norm[i, j] = np.linalg.norm(X[:, i] - X[:,j])
Norm_sort = np.argsort(Norm, axis=1)
C = Norm_sort[:, 0:c]

# Xc : N×n×c dimension matrix, T_c,H_c,Enz_c : N×c dimension matrix, Yc : N×c×1 dimension matrix
Xc = np.empty((C.shape[0], X.shape[0], C.shape[1]))
for i in range(Xc.shape[0]):
    for j in range(Xc.shape[2]):
        Xc[i, :, j] = X[:, C[i, j]]
T_c = Xc[:, 0, :]
H_c = Xc[:, 1, :]
Enz_c = Xc[:, 2, :]

yc_H = f_H(T_c,H_c,Enz_c)
Yc_H = yc_H[:, :, np.newaxis]
yc_Enz = f_Enz(T_c,H_c,Enz_c)
Yc_Enz = yc_Enz[:, :, np.newaxis]

# Phi : N×c×(n+1) dimension matrix, (Phi)'×Phi : N×2×2 dimension matrix, inverse matrix of phi : N×c×c dimension matrix, theta_ls : N×(n+1)×1 dimension matrix
one = np.ones((Xc.shape[0], 1, Xc.shape[2]))
Phi_T = np.concatenate((Xc, one), axis=1)
Phi = Phi_T.transpose(0, 2, 1)
phi = Phi_T @ Phi
inv_phi = np.linalg.inv(phi)
PHI = inv_phi @ Phi_T
Theta_ls_H = PHI @ Yc_H
Theta_ls_Enz = PHI @ Yc_Enz