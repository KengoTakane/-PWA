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
N = 5
c = 5
s = 5
num_piece = int(s*(s-1)/2)

Ta = np.random.uniform(T_min,T_max,(N,))
H = np.random.uniform(H_min,H_max,(N,))
Enz = np.random.uniform(Enz_min,Enz_max,(N,))

A = np.array([-0.06931422,  0.00707421, -0.02392939, -1.71191157, -0.32115151])
B = np.array([-0.03398082, -0.78576906, -0.01510164, -0.60605515,  0.16966456])
C = np.array([-1.34049162, -12.69764814,  -0.26775359,  -8.89824217, -1.1117551])
D = np.array([385.15248942, 3764.67933988,   76.38631781, 2710.63059713,  317.64563539])
Q = np.array([-0.30835227,  0.05842341, -0.05006525, -0.1019953,   0.01096989,  0.33793413, 0.0194443, -0.02308869,  0.07820222, -0.37282347])
T = np.array([0.05622362, -0.02756808,  0.07751531, -0.07667624,  0.03930941,  0.06601659, 0.05034097,  0.16239576,  0.09710297, -0.1752447])
R = np.array([-1.32558496,  0.93937784, -0.73272573, -0.40374013,  0.38693289,  0.31749014, 0.87706155, -0.64391473, -0.67765675,  0.68098752])
S = np.array([400.45264827, -272.53657897,  219.31059418,  127.70309902, -113.19684429, -102.87765568, -251.76944887,  188.89971037,  198.86975282, -186.53938676])


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


def fun_value(q,T,Rh):
    if state_partion(q,T,Rh) == 0:
        e = (fun(q,T,Rh)-fun0(q,T,Rh))**2
        er = np.sqrt(e)
        print('area 0, fun0()=',fun0(q,T,Rh))
    elif state_partion(q,T,Rh) == 1:
        e = (fun(q,T,Rh)-fun1(q,T,Rh))**2
        er = np.sqrt(e)
        print('area 1, fun1()=',fun1(q,T,Rh))
    elif state_partion(q,T,Rh) == 2:
        e = (fun(q,T,Rh)-fun2(q,T,Rh))**2
        er = np.sqrt(e)
        print('area 2, fun2()=',fun2(q,T,Rh))
    elif state_partion(q,T,Rh) == 3:
        e = (fun(q,T,Rh)-fun3(q,T,Rh))**2
        er = np.sqrt(e)
        print('area 3, fun3()=',fun3(q,T,Rh))
    elif state_partion(q,T,Rh) == 4:
        e = (fun(q,T,Rh)-fun4(q,T,Rh))**2
        er = np.sqrt(e)
        print('area 4, fun4()=',fun4(q,T,Rh))
    else:
        print("error")


print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("error : non-linear - linear")
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
for i in range(H.shape[0]):
    for j in range(Enz.shape[0]):
        for k in range(Ta.shape[0]):
            error(H[i],Enz[j],Ta[k])


print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")

print("value of linear function")
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------")
for i in range(H.shape[0]):
    for j in range(Enz.shape[0]):
        for k in range(Ta.shape[0]):
            fun_value(H[i],Enz[j],Ta[k])

