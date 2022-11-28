from cProfile import label
from re import T
import numpy as np
import statsmodels.api as sm
from scipy import linalg, optimize
from sklearn.svm import SVC
import matplotlib.pyplot as plt
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


def H(t,T):
    return H_plusinf + (H_minusinf-H_plusinf)/(1+np.exp((k(T)*t)*(H_minusinf-H_plusinf))*(H_minusinf-H_0)/(H_0-H_plusinf))


# H()のグラフ化
sns.set()
x = np.linspace(0,20,100)
fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.plot(x,H(x,12+273),label="$12^{\circ}C$")
ax.plot(x,H(x,15+273),label="$15^{\circ}C$")
ax.plot(x,H(x,18+273),label="$18^{\circ}C$")
ax.set_xlabel('time(days)')
ax.set_ylabel('Hue(${}^\circ$)')
ax.set_title('Quest')
plt.legend(loc='best')
sns.despine()


Tem = np.linspace(10+273,20+273,100)
a = (k(20+273)-k(10+273))/(20-10)
b = k(20+273)-a*(20+273)
fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot(Tem,k(Tem),label="$k(T)$")
ax.plot(Tem,a*Tem+b,label="$k_{appro}(T)$")
ax.set_xlabel('Temperature(K)')
ax.set_ylabel('$k(T)$')
plt.legend(loc='best')


fig3 = plt.figure()
ax = fig3.add_subplot(111)
ax.plot(Tem,H(1,Tem),label="$t=1$")
ax.plot(Tem,H(5,Tem),label="$t=5$")
ax.plot(Tem,H(10,Tem),label="$t=10$")
ax.set_xlabel('Tempareture(K)')
ax.set_ylabel('Hue(${}^\circ$)')
ax.set_title('Quest')
plt.legend(loc='best')
sns.despine()

plt.show()
