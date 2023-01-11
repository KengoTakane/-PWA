import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

H_0 = 62.9
H_plusinf = 43.1
H_minusinf = 124
k_ref = 0.0019
E_a = 170.604
T_ref = 288.15
Rg = 0.008314

Enz_0 = 61

def k(T):
    return k_ref*np.exp((E_a/Rg)*(1/T_ref-1/T))

def H(t,T): # 解
    return H_plusinf + (H_minusinf-H_plusinf)/(1+np.exp((k(T)*t)*(H_minusinf-H_plusinf))*(H_minusinf-H_0)/(H_0-H_plusinf))

init   = [H_0-H_plusinf, Enz_0]
t_span = [0.0,22.0]
t_eval = np.linspace(*t_span,300) # time for sampling

def fun(t,X,T): # 微分方程式
    H,Enz = X
    return [-k(T)*H*Enz, k(T)*H*Enz]


sol_12 = solve_ivp(fun,t_span,init,method='RK45',t_eval=t_eval,args=[12+273])
sol_15 = solve_ivp(fun,t_span,init,method='RK45',t_eval=t_eval,args=[15+273])
sol_18 = solve_ivp(fun,t_span,init,method='RK45',t_eval=t_eval,args=[18+273])

print('sol.y shape: ', sol_15.y.shape)
print('H(0): ', init[0]+H_plusinf)
print('Enz(0): ', init[1])
# print('H(22): ', sol.y[0,:])
# print('Enz(22): ', sol.y[1,:])

sns.set()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

x = np.linspace(0,22,100)
ax1.plot(sol_15.t[:],sol_15.y[0,:]+H_plusinf, label="$H(t)$ from DeEq")
ax1.plot(x,H(x,15+273),label="$H(t)$ from Sol")
ax1.set_ylim(40, 70)
ax1.set_xlabel('time(days)')
ax1.set_ylabel('Hue(${}^\circ$)')
ax1.set_title('$H(t), T=15^{\circ}C$')
ax1.legend(loc='upper right')

"""
ax2.plot(sol_18.t[:],sol_18.y[0,:]+H_plusinf, label="$H(t)$ from DeEq")
ax2.plot(x,H(x,18+273),label="$H(t)$ from Sol")
ax2.set_ylim(40, 70)
ax2.set_xlabel('time(days)')
ax2.set_ylabel('Hue(${}^\circ$)')
ax2.set_title('$H(t),  T=18^{\circ}C$')
ax2.legend(loc='upper right')

"""
ax2.plot(sol_15.t[:], sol_15.y[1,:], label="$Enz(t)$")
ax2.set_xlabel('time(days)')
ax2.set_ylabel('Hue(${}^\circ$)')
ax2.set_title('Enz(t),  T=15^{\circ}C')
ax2.legend(loc='lower right')

plt.show()
