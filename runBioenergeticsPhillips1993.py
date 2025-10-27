'''
This is a code to simulate bioenergetics of muscle contraction
In particular, we limit the modelling to the state of PCr within the muscle, 
and prescribe the use of ATP. Thus, we are not simulating the full contraction 
within this code 

This version optimizes the model parameters to reproduce observed experimental results

Ryan Konno
School of Biomedical Sciences 
The University of Queensland
'''

# Import 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cmap
import itertools
import sys 
sys.path.append('./')

# Parameters 
params = {
    # Time parameters
    't_start': 0, # s
    't_end': 90, # s, 

    # Energetics parameters 
    'Energetics_model':{
        # Define parameters for the ODE 
        # .... In this code they are used as initial values for the optimization
        'k_recpcr': 0.0214, # 1/s
        'k_m_1':  0.0367, # 1/s

        # Energetic constant to predict energetic rates
        # 'r_r': 522e-3, # J umol^-1, Calculated based on the glycogen and atp enthalpys
        'r_r': 2802e-3 # J umol^{-1}, Based on enthalpy of glucose catabolism (Kabo et al. 2013)
        # 'r_r': 0.04489659, # J umol^-1, Optimized to experimental data (Phillips et al. 1993)
    }
}

#### 
# Import experimental data to fit the model
#       NOTE: Experimental data only had a 4s time resolution
datapath = 'Data/PhillipsData_dPCR.csv'
exp_data = np.loadtxt(datapath, delimiter=",", skiprows=1)
# Scale the data so that pcr is negative 
t_exp = exp_data[:,0] # Units of s
pcr_exp = -exp_data[:,1] # Converted to units of umol/(g wet wt)

# Import the experimental data for the heat rates
datapath = 'Data/PhillipsData_Heat.csv'
exp_data_heat = np.loadtxt(datapath, delimiter=",", skiprows=1)
t_exp_heat = exp_data_heat[:,0] # Units of s 
Q_r_exp = exp_data_heat[:,1] / 5 * 1e-3 # Converted to units of W/g

### 
# Create a class for the ode solver 
class Bioenergetics(): 
    def __init__(self, k_recpcr, k_m): 
        # Initialize the main parameters
        self.k_recpcr = k_recpcr
        self.k_m =  k_m 

    # Compute the recovery of ATP 
    def pcrRecoveryRate(self, t, y): 
        '''
        Function of time and current pcr concentration 
        '''
        pcr, Ma = y

        # New model 2a
        # k_recpcr = params['Energetics_model']['k_recpcr']
        # k_m =  params['Energetics_model']['k_m_1'] # Free parameter (fit to exp data)
        # print(f'Parameters: k_recpcr = {self.k_recpcr}, k_m_1 = {self.k_m}')

        dpcrdt = - self.ATP(t) + self.k_recpcr * Ma * (pcr)
        dMadt = self.k_m * (pcr - Ma)

        # Return rate of recovery 
        return (dpcrdt, dMadt)

    def ATP(self, t): 
        ''' 
        Function defining ATP use during the contraction 
        
        We assume a constant stimulation of 12s 
        '''
        atp_peak = 0.213 # umol/s/(g wet wt) [Based on initial rates from Phillips et al. 1993]
        tstimend = 12
        trampend = 1
        return atp_peak * (t < tstimend) + atp_peak * 0.5 * (np.sin((np.pi*(t-tstimend) / (trampend-tstimend) + np.pi/2)) + 1) * (t > tstimend) * (t < trampend)
        # return 50 * (t%10 < 5) 

    def solveODE(self, t_start, t_end):
        '''
        Function to solve the ode 
        '''
        # Initial conditions for ode
        PCr_initial = 0.0  # Initial d[PCr] in mM
        activation_initial = 0.0  # Initial mitochondrial activation
        y0 = [PCr_initial, activation_initial]

        # Solve the ode 
        from scipy.integrate import odeint, solve_ivp
        tspan = (t_start, t_end)

        # Solve the IVP 
        sol = solve_ivp(self.pcrRecoveryRate, tspan, y0, max_step = 0.01)

        return sol 
    
    def computeEnergetics(self, pcr, Ma, r_r): 
        '''
        Function to compute the energetic rate of the recovery process 
        '''
        return r_r * self.k_recpcr * Ma * pcr

# Run Bioenergetics model 
model = Bioenergetics(params['Energetics_model']['k_recpcr'], params['Energetics_model']['k_m_1'])
sol = model.solveODE(params['t_start'], params['t_end']) # Solve the IVP 
PCr, activation = sol.y
Q_r = model.computeEnergetics(PCr, activation, params['Energetics_model']['r_r']) # Compute the recovery energetic rate

# # Determine the constant r_r via optimization 
# def f_fit(t, params): 
#     r_r = params
#     PCr_ = np.interp(t, sol.t, PCr)
#     activation_ = np.interp(t, sol.t, activation)
#     return model.computeEnergetics(PCr_, activation_, r_r)
# from scipy.optimize import curve_fit 
# popt = curve_fit(f_fit, t_exp_heat, Q_r_exp, p0 = params['Energetics_model']['r_r']) 
# print(f'r_r = {popt}')
# Q_r = model.computeEnergetics(PCr, activation, popt[0]) # Compute the recovery energetic rate


# Plot the comparison between PCr experimental and modelled 
fig, ax  = plt.subplots(figsize = (6,4), layout = 'constrained') 
ax.plot(sol.t, PCr, label = 'PCr (model)', color = 'k')
ax.plot(t_exp, pcr_exp, label = 'PCr (Phillip et al. 1993)', ls = 'None', marker = '.', color = 'k')
ax.set_xlabel('Time (s)')
ax.set_ylabel('PCr Concentration [umol/g]')
plt.legend()
plt.savefig('./Figures/PCrParamOpt.pdf')

# Plot the energetic rates (experimental and predicted rates) 
fig, ax = plt.subplots(figsize = (6,4), layout = 'constrained')
ax.plot(sol.t, Q_r, label = '\dot Q_r (model)', color = 'k')
ax.plot(t_exp_heat, Q_r_exp, label = '\dot Q_r (exp)', color = 'k', ls = 'dashed') 
ax.plot((0,9), (12.8e-3,12.8e-3), label = '\dot Q_i (exp)', color = 'k', ls = 'dashed') 
ax.set_xlabel('Time (s)')
ax.set_ylabel('Recovery energetic rate (W/g)')
plt.legend()
plt.savefig('./Figures/RecoveryEnergeticRate.pdf')

# Total energetic rates (experimental and predicted rates) 
fig, ax = plt.subplots(figsize = (6,4), layout = 'constrained')
ax.plot(sol.t, Q_r, label = '\dot Q_r (model)', color = 'k')
ax.plot(t_exp_heat, Q_r_exp, label = '\dot Q_r (exp)', color = 'k', ls = 'dashed') 
ax.set_xlabel('Time (s)')
ax.set_ylabel('Total energetic rate (W/g)')
plt.legend()
plt.savefig('./Figures/RecoveryOnlyEnergeticRate.pdf')



# # Plot the output
# fig, axs =plt.subplots(1, 2, figsize = (12,6), layout = 'constrained')

# ax = axs[0]
# ax.plot(sol.t, PCr, label='[PCr]', color = 'blue')
# ax.plot(sol.t, activation, label='Mitochondrial Activation', color='red', linestyle=':')
# ax.plot(sol.t, ATP(sol.t), label='[ATP used]', color='green')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Concentration (mM)')
# ax.grid()

# ax = axs[1]
# # Compute the rates 
# pcr_rate, activation_rate = pcrRecoveryRate(sol.t, sol.y)
# ax.plot(sol.t, pcr_rate, label='[PCr]', color='blue')
# ax.plot(sol.t, activation_rate, label='Mitochondrial Activation', color='red', linestyle=':')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Rates (mM/s)')
# ax.legend()
# ax.grid()

# plt.savefig('Figures/ATPRegenEnergetics.jpg')
plt.show()

