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
        'k_recpcr': 0.05,
        'k_m_1':  0.2, # Free parameter (fit to exp data)

        # Energetic costant to predict energetic rates
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

def f_opt(x):
    '''
    Function to optimize
        Input: 
            x - parameter vector 
    '''

    ####
    # Define the bioenergetics model 
    model = Bioenergetics(x[0] , x[1])

    #####
    # Solve the ode 
    sol = model.solveODE(params['t_start'], params['t_end'])

    # PCr, ADP, activation = sol.T  # Transpose to get individual variables
    PCr, activation = sol.y  # Transpose to get individual variables

    ####
    # Error calculation 
    # Interpolate values to compare 
    pcr_model = np.interp(t_exp, sol.t, PCr)

    # Plot the comparison between PCr experimental and modelled 
    # fig, ax  = plt.subplots(figsize = (6,4), layout = 'constrained') 
    # ax.plot(sol.t, PCr, label = 'PCr (model)')
    # ax.plot(t_exp, pcr_exp, label = 'PCr (model)', ls = 'None', marker = '.')
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('PCr Concentration [mol/g]')
    # plt.legend()
    # plt.show()

    # Compute the error 
    error = np.linalg.norm(pcr_exp - pcr_model) / np.linalg.norm(pcr_exp)
    return error 


# Perform optimization 
from scipy.optimize import minimize
x0 = (0.6, params['Energetics_model']['k_m_1'])
constraints = ((0,10), (0,10))
opt_res = minimize(f_opt, x0, method = 'Nelder-Mead', bounds = constraints, options = {'disp': True, 'maxiter': 500})
print(f'Optimal parameters: {opt_res.x}')

# Rerun with optimal parameters 
model = Bioenergetics(*opt_res.x)
sol = model.solveODE(params['t_start'], params['t_end']) # Solve the IVP 
PCr, activation = sol.y

# Plot the comparison between PCr experimental and modelled 
fig, ax  = plt.subplots(figsize = (6,4), layout = 'constrained') 
ax.plot(sol.t, PCr, label = 'PCr (model)', color = 'k')
ax.plot(t_exp, pcr_exp, label = 'PCr (Phillip et al. 1993)', ls = 'None', marker = '.', color = 'k')
ax.set_xlabel('Time (s)')
ax.set_ylabel('PCr Concentration [mol/g]')
plt.legend()
plt.savefig('./Figures/PCrParamOpt.pdf')



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

