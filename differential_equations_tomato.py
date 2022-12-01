import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
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


def f(t, v): # H:v[0], Enz:v[1]
    return [-k(15+273)*v[0]*v[1], k(15+273)*v[0]*v[1]]

v0 = [65, 10]

solver = ode(f)
solver.set_integrator(name="dop853")
solver.set_initial_value(v0)

tw = 25
dt = tw / 1000
t = 0.0
ts = []
Hs = []
Enzs = []

while solver.t < tw:
    solver.integrate(solver.t+dt)
    ts += [solver.t]
    Hs += [solver.y[0]]
    Enzs += [solver.y[1]]

sns.set()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(ts, Hs, label="$H(t)$")
ax1.set_xlabel('time(days)')
ax1.set_ylabel('Hue(${}^\circ$)')
ax1.set_title('$H(t)$')

ax2.plot(ts, Enzs, label="$Enz(t)$")
ax2.set_xlabel('time(days)')
ax2.set_ylabel('Hue(${}^\circ$)')
ax2.set_title('Enz(t)')

plt.legend(loc='best')
plt.show()
