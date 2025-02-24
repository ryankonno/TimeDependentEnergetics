'''
This code is designed to run the time dependent energetics for a single MU

Ryan Konno
School of Biomedical Sciences 
The University of Queensland
'''

# Import 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
sys.path.append('./')

# Parameters 
params = {
    't_start': 0, # s
    't_end': 0.2, # s
    'N_t': 1000, # int, Number of time steps (activation model)

    'f_stim': 20, # Hz, Stimulation frequency

    'N_MU_sim': 1, # int, Number of MUs to stimulate    

    # Activation model parameters
    'Activation_model': {
        # Time constant for excitation-activation dynamics 
        # Slow properties 
        "Tau_1": 0.0422, # Assume constant value from MCL (2023)
        # "Tau_2": 0.256, # Scaling based on MCL (2023)
        "Tau_2": 0.1,
        # Fast properties 
        # "Tau_1": 0.0422, # Very little change between fibre type - assume constant (BH, 2003)
        # "Tau_2": 0.256/2, # Decay constant for fast twitch Fibres assuming 1/2 rate (Baylor and Hollingworth, 2003)

        "K": 0.1025,
        "n": 3, # Hill coefficient
    },

    # Energetics parameters 
    'Energetics_model':{
        'nu': 1.72,

        'H_PCr': 36e3, # J/mol, Enthalpy of pcr

        'r_am': 0.6177, # W, Maximum heat rate of isometric contraction

    }
}

# Define the time trace 
t = np.linspace(params['t_start'], params['t_end'], params['N_t'])

#____________________________
# Stimulation...

stim_inter = 1/params['f_stim'] # Define distance between stimulations

# Determine the times at which stimulations occur
t_stim = np.arange(t[0], t[-1] + stim_inter, stim_inter)

# Find the indices in the time vector closest to stimulation times
t_stim_idxs = np.searchsorted(t, t_stim, side='left')

######################################################
'''
Run the activation model 
'''

#  Import the activation model 
from Models.MUActivationModel import ActivationModel

# Initialize 
act_model = ActivationModel(params['Activation_model'], t, True)

# Run the Ca dynamics
stim_vec, ca_vec, catn_vec = act_model.runExcAct(t_stim_idxs)

# Plot Ca dynamics
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t, stim_vec, label = 'Stimulus')
ax.plot(t, ca_vec, label = 'Ca')
ax.plot(t, catn_vec, label = 'CaTn')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Normalized values')
ax.legend()

######################################################
'''
Run the energetics calculation based on the activation levels
'''

# Import the model 
from Models.MUEnergeticsModel import EnergeticsModel

# Initialize energetics model 
energy_model = EnergeticsModel()

# Compute the energetics based on Ca and CaTn 
q_a, q_m = energy_model.actEnergetics(t, ca_vec, catn_vec, params['Energetics_model'])

fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t, q_a, label = 'Activation heat') 
ax.plot(t, q_m, label = 'Maintenance heat')
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Energetic rate (1/s)')

plt.show()



