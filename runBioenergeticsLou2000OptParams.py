'''
This is a code to simulate bioenergetics of muscle contraction
Within this code the modelling is limited to the amount of PCr within the muscle, 
based on a prescribed use of ATP. 
The full contraction is not simulated within this code 

This version is used to compare model predictions to those from Barclay 1995. 

Ryan Konno
School of Biomedical Sciences 
The University of Queensland
'''

# Import 
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 14
import matplotlib.cm as cmap
import sys 
sys.path.append('./')

# Parameters 
params = {
    # Time parameters
    't_start': 0, # s
    't_end': 150, # s, 30 cycles at 5 seconds each 

    # Energetics parameters 
    'Energetics_model':{
        # Define parameters for the ODE 
        # .... In this code they are used as initial values for the optimization
        'k_recpcr': 0.0214, # 1/s
        'k_m_1':  0.0367, # 1/s

        # Energetic constant to predict energetic rates
        # 'r_r': 522e-3, # J umol^-1, Calculated based on the glycogen and atp enthalpys

        # Optimised to Barclay et al. 1995 dataset
        'r_r': 0.139, # J umol^-1, SOL, Based on ratio in initial heat to recovery heat after 30 cycles compared to Barclay et al. 1995
        'r_r': 0.0403, # J umol^-1, EDL, Based on ratio in initial heat to recovery heat after 30 cycles compared to Barclay et al. 1995

        'r_r': 0.04489659, # J umol^-1, Optimized to experimental data (Phillips et al. 1993)
    }
}

### 
# Create a class for the ode solver 
class Bioenergetics(): 
    def __init__(self, k_recpcr, k_m): 
        # Initialize the main parameters
        self.k_recpcr = k_recpcr
        self.k_m =  k_m 

        # Counter for the cycles 
        self.cycle_count = 1 # NOTE: starts at 1

    # Compute the recovery of ATP 
    def pcrRecoveryRate(self, t, y): 
        '''
        Function of time and current pcr concentration 
        '''
        pcr, Ma = y

        # New model 2a
        dpcrdt = - self.ATP(t) + self.k_recpcr * Ma * (pcr)
        dMadt = self.k_m * (pcr - Ma)

        # Return rate of recovery 
        return (dpcrdt, dMadt)

    def ATP(self, t): 
        ''' 
        Function defining ATP use during the contraction 
        
        Modified to simulate the contractions from Lou et al. 2000
        '''
        # Constant atp_peak [we use variable atp usage now... see below]
        # atp_peak = 0.75 # umol/s/(g wet wt) [computed using computeParametersBarclay1995.py]

        tstimend = 6.7 # s, Length of stimulation (Lou et al. 2000)

        # parameters 
        dHdt_i = 0.658e-3 / tstimend # J/s, 
        mass = 18.3e-3 # g, From Curtin et al. 1997
        G_atp = 60e3 # J/mol 

        atp_peak = dHdt_i / G_atp * 1e6 / mass # umol/s/(g wet wt) [computed using computeParametersBarclay1995.py]

        return atp_peak * (t < tstimend)

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

# Import experimental data  
ratio = 1.708

# Optimise r_r to the soleus 
def opt_fun(x): 
    Q_r_ = model.computeEnergetics(PCr, activation, x)
    # Q_r = np.mean(Q_r_[sol.t > 145])

    Q_init = 0.658e-3  # initial heat from Lou et al. 2000

    # To compare to data we want to compare the integral of the heat rate (ie the absolute heat)
    Q_r = np.trapz(Q_r_, sol.t)

    return np.abs(Q_r / Q_init - ratio)
from scipy.optimize import minimize_scalar
opt_sol = minimize_scalar(opt_fun, (0.001,10)) 

# Update the recovery heat rate
params['Energetics_model']['r_r'] = opt_sol.x
print(f'Optimal r_r value based on ratio: {opt_sol.x}')


Q_r = model.computeEnergetics(PCr, activation, params['Energetics_model']['r_r']) # Compute the recovery energetic rate

# Optimise the model recovery ratio to the dataset from 

# Plot the comparison between PCr experimental and modelled 
fig, ax  = plt.subplots(figsize = (6,4), layout = 'constrained') 
ax.plot(sol.t, PCr, label = 'PCr (model)', color = 'k')
# ax.plot(t_exp, pcr_exp, label = 'PCr (Phillip et al. 1993)', ls = 'None', marker = '.', color = 'k')
ax.set_xlabel('Time (s)')
ax.set_ylabel('PCr Concentration [umol/g]')
plt.legend()
# plt.savefig('./Figures/B1995PCr.pdf')

# Plot the energetic rates (experimental and predicted rates) 
fig, ax = plt.subplots(figsize = (6,4), layout = 'constrained')
ax.plot(sol.t, Q_r, label = '\dot Q_r (model)', color = 'k')
ax.plot(sol.t, model.ATP(sol.t) * 60e3 / 10**6, label = '\dot Q_i', color = 'k', ls = 'dashed') 
ax.set_xlabel('Time (s)')
ax.set_ylabel('Recovery energetic rate (W/g)')
plt.legend()
# plt.savefig('./Figures/B1995Energetics.pdf')


plt.show()

