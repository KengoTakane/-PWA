import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg, optimize
from scipy.integrate import solve_ivp

a = 1.00
b = -1.59
E = 0.045
Rg = 0.008314
n = 3
N = 20
c = 5
s = 5
T = np.random.uniform(278,298,(N,))
Rh = np.random.uniform(30,95,(N,))
q = np.random.uniform(900,1100,(N,))


def fun(q,T,Rh):
    return ((-a * np.exp(b*Rh/100) * np.exp(-E/(Rg*T)))/1000) * q

def fun0(q,T,Rh):
    return (-4.21270271e-04)*q + (-4.84453301e-05)*T + (6.47483981e-03)*Rh + (-3.29723012e-01)
def fun1(q,T,Rh):
    return (-2.36178784e-04)*q + (-3.14845920e-05)*T + (3.51970772e-03)*Rh + (-3.06409958e-01)
def fun2(q,T,Rh):
    return (-3.22164500e-04)*q + (-5.36849855e-05)*T + (5.42080197e-03)*Rh + (-3.63390858e-01)
def fun3(q,T,Rh):
    return (-2.52650737e-04)*q + (-1.67910305e-05)*T + (4.06639287e-03)*Rh + (-3.42315514e-01)
def fun4(q,T,Rh):
    return (-5.55708636e-04)*q + (1.35079377e-05)*T + (8.64274190e-03)*Rh + (-3.13246241e-01)

def state_partion(q,T,Rh):
    if (0.02627317)*q+(-0.07538652)*T+(-0.16798629)*Rh+(8.4387248)>=0 and (-0.00111771)*q+(0.04318842)*T+(-0.24951434)*Rh+(3.37998739)>=0 and (0.01880834)*q+(-0.04245705)*T+(-0.34649457)*Rh+(16.72282594)>=0 and (-0.01876554)*q+(-0.05633143)*T+(0.24132598)*Rh+(24.14807339)>=0:
        return 0
    elif (0.02627317)*q+(-0.07538652)*T+(-0.16798629)*Rh+(8.4387248)<=0 and (-0.04472564)*q+(0.05458913)*T+(0.2253457)*Rh+(9.62861827)>=0 and (-0.03862955)*q+(0.0601666)*T+(0.20345188)*Rh+(1.78944618)>=0 and (-0.05427779)*q+(-0.03055113)*T+(0.14436165)*Rh+(51.78167187)>=0:
        return 1
    elif (-0.00111771)*q+(0.04318842)*T+(-0.24951434)*Rh+(3.37998739)<=0 and (-0.04472564)*q+(0.05458913)*T+(0.2253457)*Rh+(9.62861827)<=0 and (0.02798923)*q+(-0.01195942)*T+(-0.23591518)*Rh+(-6.80356885)>=0 and (-0.00845947)*q+(-0.00280735)*T+(0.32142254)*Rh+(-7.37417206)>=0:
        return 2
    elif (0.01880834)*q+(-0.04245705)*T+(-0.34649457)*Rh+(16.72282594)<=0 and (-0.03862955)*q+(0.0601666)*T+(0.20345188)*Rh+(1.78944618)<=0 and (0.02798923)*q+(-0.01195942)*T+(-0.23591518)*Rh+(-6.80356885)<=0 and (-0.01385068)*q+(0.02021543)*T+(0.1428877)*Rh+(-0.34880202)>=0:
        return 3
    elif (-0.01876554)*q+(-0.05633143)*T+(0.24132598)*Rh+(24.14807339)<=0 and (-0.05427779)*q+(-0.03055113)*T+(0.14436165)*Rh+(51.78167187)<=0 and (-0.00845947)*q+(-0.00280735)*T+(0.32142254)*Rh+(-7.37417206)<=0 and (-0.01385068)*q+(0.02021543)*T+(0.1428877)*Rh+(-0.34880202)<=0:
        return 4
    else:
        return 5


def error(q,T,Rh):
    if state_partion(q,T,Rh) == 0:
        e = (fun(q,T,Rh)-fun0(q,T,Rh))**2
        er = np.sqrt(e)
        print(er)
    elif state_partion(q,T,Rh) == 1:
        e = (fun(q,T,Rh)-fun1(q,T,Rh))**2
        er = np.sqrt(e)
        print(er)
    elif state_partion(q,T,Rh) == 2:
        e = (fun(q,T,Rh)-fun2(q,T,Rh))**2
        er = np.sqrt(e)
        print(er)
    elif state_partion(q,T,Rh) == 3:
        e = (fun(q,T,Rh)-fun3(q,T,Rh))**2
        er = np.sqrt(e)
        print(er)
    elif state_partion(q,T,Rh) == 4:
        e = (fun(q,T,Rh)-fun4(q,T,Rh))**2
        er = np.sqrt(e)
        print(er)
    else:
        print("error")

        
def fun_value(q,T,Rh):
    if state_partion(q,T,Rh) == 0:
        e = (fun(q,T,Rh)-fun0(q,T,Rh))**2
        er = np.sqrt(e)
        print(fun(q,T,Rh))
    elif state_partion(q,T,Rh) == 1:
        e = (fun(q,T,Rh)-fun1(q,T,Rh))**2
        er = np.sqrt(e)
        print(fun(q,T,Rh))
    elif state_partion(q,T,Rh) == 2:
        e = (fun(q,T,Rh)-fun2(q,T,Rh))**2
        er = np.sqrt(e)
        print(fun(q,T,Rh))
    elif state_partion(q,T,Rh) == 3:
        e = (fun(q,T,Rh)-fun3(q,T,Rh))**2
        er = np.sqrt(e)
        print(fun(q,T,Rh))
    elif state_partion(q,T,Rh) == 4:
        e = (fun(q,T,Rh)-fun4(q,T,Rh))**2
        er = np.sqrt(e)
        print(fun(q,T,Rh))
    else:
        print("error")

print("error : non-linear - linear")
for i in range(q.shape[0]):
    for j in range(T.shape[0]):
        for k in range(Rh.shape[0]):
            error(q[i],T[j],Rh[k])

print("----------------------------------------------------------------------------------------")
print("value of non-linear function")
for i in range(q.shape[0]):
    for j in range(T.shape[0]):
        for k in range(Rh.shape[0]):
            fun_value(q[i],T[j],Rh[k])