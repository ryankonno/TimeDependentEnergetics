'''
Code to compute the parameters used based on the Barclay 1995 study 
'''

import numpy as np 

# Constants from Barclay 1995
sigma_0 = 2e5         # Pa 
mass = 0.51e-3        # g 
rho = 1e6             # g/m^3 
l_0 = 9.7e-3          # m 

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
dATPdt_vec = np.array((0.225, 0.176, 0.166, 0.164)) * F_0 * l_0 / mass  / G_atp * 10**6 # umol/g

# Use a cubic spline to interpolate these values 
# from scipy.interpolate import interp1d 
# cs = interp1d(cycles, dATPdt_vec, '')

# Use an exponential fit 
def f(x, a, b, c, d): 
    return a * np.exp(b * x - c) + d
from scipy.optimize import curve_fit 
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