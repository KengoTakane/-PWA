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
N = 10
c = 5
s = 5
num_piece = int(s*(s-1)/2)

Ta = np.random.uniform(T_min,T_max,(N,))
H = np.random.uniform(H_min,H_max,(N,))
Enz = np.random.uniform(Enz_min,Enz_max,(N,))

A = np.array([-0.01999273, -0.03169743, -1.20410036, -0.24750489, -0.22860312])
B = np.array([-0.01252207, -0.05697102, -0.52635242,  0.08347816,  0.12462958])
C = np.array([-0.2519847,   -0.36334543, -10.71015622,  -1.26621934,  -2.00428928])
D = np.array([71.5492702,   106.47446482, 3222.53507534,  364.91430271,  573.79244091])
Q = np.array([-0.11084162, -0.03725109, -0.07088854, -0.22620689, -0.04615925,  0.06724736,
 -0.09186333,  0.06621457,  0.19608856, -0.0676158])
T = np.array([0.05622362, -0.02756808,  0.07751531, -0.07667624,  0.03930941,  0.06601659,
  0.05034097,  0.16239576,  0.09710297, -0.1752447])
R = np.array([-0.66307958, -0.30551776, -1.18766009, -1.89590908, -0.6266267,  -0.86662505,
 -1.41328389,  0.93071362, 1.19442485, -0.44786036])
S = np.array([189.44100064,   92.07421735,  335.83591432,  558.39828918,  181.03121757,
  239.57974671,  406.03622232, -286.27497866, -367.66327063,  144.94121378])


############################################################
#########################非線形関数#########################
############################################################

def k_rate(Ta):
    return k_ref*np.exp((E_a/Rg)*(1/T_ref-1/Ta))

def fun(H,Enz,Ta):
    return -k_rate(Ta)*H*Enz


############################################################
####################区分アフィン関数#########################
############################################################

def fun0(H,Enz,Ta):
    return A[0]*H + B[0]*Enz + C[0]*Ta + D[0]
def fun1(H,Enz,Ta):
    return A[1]*H + B[1]*Enz + C[1]*Ta + D[1]
def fun2(H,Enz,Ta):
    return A[2]*H + B[2]*Enz + C[2]*Ta + D[2]
def fun3(H,Enz,Ta):
    return A[3]*H + B[3]*Enz + C[3]*Ta + D[3]
def fun4(H,Enz,Ta):
    return A[4]*H + B[4]*Enz + C[4]*Ta + D[4]

def state_partion(H,Enz,Ta):
    if Q[0]*H+T[0]*Enz+R[0]*Ta >=0 and Q[1]*H+T[1]*Enz+R[1]*Ta >=0 and Q[2]*H+T[2]*Enz+R[2]*Ta >=0 and Q[3]*H+T[3]*Enz+R[3]*Ta >=0:
        return 0
    elif Q[0]*H+T[0]*Enz+R[0]*Ta <=0 and Q[4]*H+T[4]*Enz+R[4]*Ta >=0 and Q[5]*H+T[5]*Enz+R[5]*Ta >=0 and Q[6]*H+T[6]*Enz+R[6]*Ta >=0:
        return 1
    elif Q[1]*H+T[1]*Enz+R[1]*Ta <=0 and Q[4]*H+T[4]*Enz+R[4]*Ta <=0 and Q[7]*H+T[7]*Enz+R[7]*Ta >=0 and Q[8]*H+T[8]*Enz+R[8]*Ta >=0:
        return 2
    elif Q[2]*H+T[2]*Enz+R[2]*Ta <=0 and Q[5]*H+T[5]*Enz+R[5]*Ta <=0 and Q[7]*H+T[7]*Enz+R[7]*Ta <=0 and Q[9]*H+T[9]*Enz+R[9]*Ta >=0:
        return 3
    elif Q[3]*H+T[3]*Enz+R[3]*Ta <=0 and Q[6]*H+T[6]*Enz+R[6]*Ta <=0 and Q[8]*H+T[8]*Enz+R[8]*Ta <=0 and Q[9]*H+T[9]*Enz+R[9]*Ta <=0:
        return 4
    else:
        return 5


############################################################
######################誤差を計算して出力######################
############################################################

def error(H,Enz,Ta):
    if state_partion(H,Enz,Ta) == 0:
        e = (fun(H,Enz,Ta)-fun0(H,Enz,Ta))**2
        er = np.sqrt(e)
        print(er)
    elif state_partion(H,Enz,Ta) == 1:
        e = (fun(H,Enz,Ta)-fun1(H,Enz,Ta))**2
        er = np.sqrt(e)
        print(er)
    elif state_partion(H,Enz,Ta) == 2:
        e = (fun(H,Enz,Ta)-fun2(H,Enz,Ta))**2
        er = np.sqrt(e)
        print(er)
    elif state_partion(H,Enz,Ta) == 3:
        e = (fun(H,Enz,Ta)-fun3(H,Enz,Ta))**2
        er = np.sqrt(e)
        print(er)
    elif state_partion(H,Enz,Ta) == 4:
        e = (fun(H,Enz,Ta)-fun4(H,Enz,Ta))**2
        er = np.sqrt(e)
        print(er)
    else:
        print("error")

print("error : non-linear - linear")
for i in range(H.shape[0]):
    for j in range(Enz.shape[0]):
        for k in range(Ta.shape[0]):
            error(H[i],Enz[j],Ta[k])