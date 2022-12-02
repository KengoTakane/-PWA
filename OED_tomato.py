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


def k(T):
    return k_ref*np.exp((E_a/Rg)*(1/T_ref-1/T))


init   = [21.5,90.0]
t_span = [0.0,23.0]
t_eval = np.linspace(*t_span,300) # time for sampling

def fun(t,X):
    H,Enz = X
    return [-k(15+273)*H*Enz, k(15+273)*H*Enz]

sol = solve_ivp(fun,t_span,init,method='RK45',t_eval=t_eval)

sns.set()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(sol.t[:],sol.y[0,:]+40, label="$H(t)$")
ax1.set_xlabel('time(days)')
ax1.set_ylabel('Hue(${}^\circ$)')
ax1.set_title('$H(t)$')

ax2.plot(sol.t[:], sol.y[1,:], label="$Enz(t)$")
ax2.set_xlabel('time(days)')
ax2.set_ylabel('Hue(${}^\circ$)')
ax2.set_title('Enz(t)')

plt.legend(loc='best')
plt.show()