'''
Code to compute the parameters used based on the Barclay 1995 study 
'''

import numpy as np 
from scipy.optimize import curve_fit 

# Constants from Barclay 1995
sigma_0 = 2e5         # Pa
rho = 1e6             # g/m^3

mass = 1.99e-3        # g, SOL
l_0 = 9.5e-3          # m, SOL
 
# mass = 2.85e-3        # g, EDL
# l_0 = 10e-3           # m, EDL

G_atp = 60e3          # J/mol Free energy of ATP

dHidt_f0l0 = 0.225    # W/F_0/l_0


#### 
# Convert the heat rates to W/g

# Compute CSA 
A_csa = mass / rho / l_0 # m^2 
print(f'A_csa = {A_csa} m^2')

# Compute maximum isometric force 
F_0 = sigma_0 * A_csa # N 
print(f'F_0 = {F_0} N')

# Compute the heat rate in W/g 
dHidt = dHidt_f0l0 * F_0 * l_0 / mass  # W/g
print(f'dHidt = {dHidt} W/g')

# Compute the rate of atp consumption 
dATPdt = dHidt / G_atp * 10**6 # umol/g 
print(f'dATPdt = {dATPdt} umol/g')

#######
'''
Compute the initial heat rates based on the four cycles measured during the experiment 
'''
cycles = np.array((1,5,15,30)) # Cycle number
# dATPdt_vec = np.array((0.225, 0.176, 0.166, 0.164)) * F_0 * l_0 / mass  / G_atp * 10**6 # umol/g, slow, initial 
dATPdt_vec = np.array((5.339988e-03, 5.726164e-02, 7.235249e-02, 7.009909e-02)) * F_0 * l_0 / mass  / G_atp * 10**6 # umol/g, slow, recovery
# dATPdt_vec = np.array((0.667, 0.616, 0.606,0.606)) * F_0 * l_0 / mass  / G_atp * 10**6 # umol/g, fast 
# dATPdt_vec = np.array((2.518854e-02, 6.404227e-02, 6.903081e-02, 7.313249e-02)) * F_0 * l_0 / mass  / G_atp * 10**6 # umol/g, fast, recovery

# Use a cubic spline to interpolate these values 
# from scipy.interpolate import interp1d 
# cs = interp1d(cycles, dATPdt_vec, '')

# Use an exponential fit 
def f(x, a, b, c, d): 
    # return a * np.exp( b* x - c) + d
    return a * b ** x- c
    # return 0.1 * a * b **(  x**d) - c
popt, _ = curve_fit(f, cycles, dATPdt_vec)

print(f'Constants for the function {popt}')

# Compute the values at all cycles 
cycles_full = np.arange(30) + 1
dATPdt_vec_full = f(cycles_full, *popt)

import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 14
fig,ax = plt.subplots(layout = 'constrained')
ax.plot(cycles_full, dATPdt_vec_full, label = 'Interpolation', lw = 2)
ax.plot(cycles, dATPdt_vec, label = 'Raw data (B1995)', ls = 'None', marker = '.', ms = 10)
ax.set_xlabel('Cycles')
ax.set_ylabel('ATP Consumption (umol/g)')
ax.legend()
plt.savefig('./Figures/B1995QiFit.pdf')
plt.show()