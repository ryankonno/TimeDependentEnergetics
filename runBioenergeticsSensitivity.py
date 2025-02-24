'''
This is a code to simulate bioenergetics of muscle contraction
In particular, we limit the modelling to the state of PCr within the muscle, 
and prescribe the use of ATP. Thus, we are not simulating the full contraction 
within this code 

This version runs a sensitivity analysis on the parameters


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
    't_start': 0, # s
    't_end': 40, # s
    'N_t': 100000, # int, Number of time steps (activation model)

    'f_stim': 20, # Hz, Stimulation frequency
    'N_stim': 6, # unitless, Number of stimulations

    'N_MU_sim': 1, # int, Number of MUs to stimulate    


    # Energetics parameters 
    'Energetics_model':{
        'nu': 1.72,

        'H_PCr': 36e3, # J/mol, Enthalpy of pcr

        'r_am': 0.6177, # W, Maximum heat rate of isometric contraction

        'tau_pcr_rec': 0.05, # s, PCr recovery time constant 

        'k_recpcr': 0.05,
        'k_m':  0.2, # Free parameter (fit to exp data)
        'pcr_max': 30, 
        'exp': 1, 

    }
}

# Constant ATP use 
def ATP(t): 
    atp_peak = 2
    tstimend = 12
    trampend = 1
    return atp_peak * (t < tstimend) + atp_peak * 0.5 * (np.sin((np.pi*(t-tstimend) / (trampend-tstimend) + np.pi/2)) + 1) * (t > tstimend) * (t < trampend)
    # return 50 * (t%10 < 5) 

# Compute the recovery of ATP 
def pcrRecoveryRate(t, y): 
    '''
    Function of time and current pcr concentration 
    '''

    pcr, ADP, Ma = y

    # Code from ChatGPT for model of energetics 
    # k_use = 0.02  # PCr utilization rate constant (1/s)
    # k_rec = 0.03  # PCr recovery rate constant (1/s)
    # k_delay = 0.01  # Delay rate constant for mitochondrial activation
    # PCr_max = 30.0  # Resting [PCr] in mM
    # dpcrdt = - k_use * (ATP(t) - ADP) + k_rec * Ma* (PCr_max - pcr)
    # dadpdt = k_use * (ATP(t) - ADP) - k_rec * Ma* (PCr_max - pcr)
    # dMadt = k_delay * (ADP - Ma) # Mitochondrial activation delay (linked to ADP levels)


    # Adaptations of above model 
    # Variant without ADP
    # dpcrdt = - k_use * (ATP(t) + pcr) + k_rec * Ma* (PCr_max - pcr)
    # dadpdt = 0 * t
    # dMadt = k_delay * (- pcr - Ma) # Mitochondrial activation delay (linked to ADP levels)
    # Variant with ATP(t) as a rate
    # k_use = 0.02
    # dpcrdt =  - k_use * (ATP(t) - ADP) + k_rec * Ma* (PCr_max - pcr)
    # ADP = (50 - ATP(t)) - pcr  
    # dadpdt = ATP(t) - k_rec * Ma* (PCr_max - pcr) 
    # dMadt = k_delay * (ADP - Ma) # Mitochondrial activation delay (linked to ADP levels)

    # dPCr_dt = -k_use * PCr + k_rec * Ma * (pcr_rest - PCr)
    # dADP_dt = k_use * PCr - k_rec * Ma * (pcr_rest - PCr)

    # New model 1
    # k_recpcr = 0.2
    # k_adppcr = 0.2 # Links ADP and PCR dynamics
    # k_m =  0.05 # Free parameter (fit to exp data)
    # pcr_max = 30 
    # dpcrdt = k_recpcr * Ma * (pcr_max - pcr) - k_adppcr * ADP
    # dadpdt = ATP(t) - k_adppcr * ADP
    # dMadt = k_m * (pcr_max - pcr - Ma)

    # New model 2
    k_recpcr = params['Energetics_model']['k_recpcr']
    k_m =  params['Energetics_model']['k_m'] # Free parameter (fit to exp data)
    pcr_max = params['Energetics_model']['pcr_max'] 
    dpcrdt = - ATP(t) + k_recpcr * Ma * (pcr_max - pcr)
    dadpdt = 0 * t # temporary to allow functioning
    # dMadt = k_m * (pcr_max - pcr - Ma)
    dMadt = k_m * (pcr_max - pcr - Ma) * params['Energetics_model']['exp']

    # New model 3, second order dynamics (Using adp as the derivative of Ma)
    # k_recpcr = 1
    # k_m =  1 # Free parameter (fit to exp data)
    # pcr_max = 30 
    # k_damp = 2
    # dpcrdt = - ATP(t) + k_recpcr * Ma * (pcr_max - pcr)
    # dadpdt = Ma # temporary to allow functioning
    # dMadt = -k_damp * ADP + k_m * (pcr_max - pcr - Ma)

    # New model 4, second order dynamics (Using adp as the derivative of Ma)
    # k_recpcr = 1
    # k_m =  1 # Free parameter (fit to exp data)
    # pcr_max = 30 
    # k_damp = 2
    # dpcrdt = - ATP(t) + k_recpcr * Ma * (pcr_max - pcr)
    # dadpdt = Ma # temporary to allow functioning
    # dMadt = -k_damp * Ma + k_m * (pcr_max - pcr - ADP)

    # Return rate of recovery 
    return (dpcrdt, dadpdt, dMadt)

# Initial conditions
PCr_initial = 30.0  # Initial [PCr] in mM (typical resting value)
ADP_initial = 0.0   # Initial [ADP] in mM
activation_initial = 0.0  # Initial mitochondrial activation

y0 = [PCr_initial, ADP_initial, activation_initial]

# Parameter values
paramvars_km = np.array((0.05, 0.2, 0.3, 0.4, 0.9))
parmavars_k_recpcr = np.array((0.01, 0.05, 0.1, 0.15, 0.2, 0.9))
paramvars = itertools.product(paramvars_km, parmavars_k_recpcr)

# Plot the output
fig, axs =plt.subplots(1, 2, figsize = (12,6), layout = 'constrained')
cmap_plot = cmap.viridis(np.linspace(0,1,len(paramvars_km) * len(parmavars_k_recpcr)))

ax = axs[0]

for idx, x in enumerate(paramvars):

    # Update the parameter 
    params['Energetics_model']['k_m'] = x[0]
    params['Energetics_model']['k_recpcr'] = x[1]

    from scipy.integrate import odeint, solve_ivp
    tspan = (params['t_start'], params['t_end'])

    # Solve the IVP 
    sol = solve_ivp(pcrRecoveryRate, tspan, y0, max_step = 0.01)

    # PCr, ADP, activation = sol.T  # Transpose to get individual variables
    PCr, ADP, activation = sol.y  # Transpose to get individual variables

    # Plot the output
    ax.plot(sol.t, PCr, label='[PCr]', color=cmap_plot[idx,:], linestyle='--')
    ax.plot(sol.t, activation, label='Mitochondrial Activation', color=cmap_plot[idx], linestyle=':')




# ax.plot(sol.t, ADP, label='[ADP]', color='red', linestyle='--')

ax.plot(sol.t, ATP(sol.t), label='[ATP used]', color='green')
# ax.set_title('PCr Recovery During Muscle Contraction')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Concentration (mM)')
# ax.legend()
ax.grid()

ax = axs[1]
# Compute the rates 
pcr_rate, adp_rate, activation_rate = pcrRecoveryRate(sol.t, sol.y)
ax.plot(sol.t, pcr_rate, label='[PCr]', color='blue')
ax.plot(sol.t, adp_rate, label='[ADP]', color='red', linestyle='--')
ax.plot(sol.t, activation_rate, label='Mitochondrial Activation', color='green', linestyle=':')
# ax.set_title('PCr Rates')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Rates (mM/s)')
ax.legend()
ax.grid()

# plt.savefig('Figures/ATPRegenEnergetics.jpg')
plt.show()

