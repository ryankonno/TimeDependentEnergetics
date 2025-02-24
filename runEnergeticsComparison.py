'''
Code compare single MU energetics for different MU parameters

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
    't_end': 1.5, # s
    'N_t': 100000, # int, Number of time steps (activation model)

    'f_stim': 20, # Hz, Stimulation frequency
    'N_stim': 6, # unitless, Number of stimulations

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

# Initialize figures for display 
fig_energy, axs_energy = plt.subplots(1,3, figsize = (12,6), layout = 'constrained')
fig_cadyn, axs_cadyn = plt.subplots(1,3, figsize = (12,6), layout = 'constrained')

# Define parameter set to compare 
#   Each of below arrays must be equivalent length
# f_stim = (5,10,15,20,25,30,35,40)
# N_stim = (5,5,5,5,5,5,5,5)

f_stim = np.linspace(5,40,5) 
N_stim = np.ones_like(f_stim) * 5

# Output storage 
mean_energy_rates = np.zeros_like(np.array(f_stim), dtype=float)
mean_force = np.zeros_like(np.array(f_stim), dtype=float)

for idx, (params['f_stim'], params['N_stim']) in enumerate(zip(f_stim, N_stim)): 
    #____________________________
    # Stimulation...

    stim_inter = 1/params['f_stim'] # Define distance between stimulations

    # Determine the times at which stimulations occur
    #   NOTE: requires 1s sti
    t_stim = np.arange(t[0],  t[-1] + stim_inter, stim_inter) 
    t_stim = t_stim[:int(params['N_stim'])]

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
    ax = axs_cadyn[0]
    ax.plot(t, ca_vec, label = 'f_stim = ' + str(params['f_stim']))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Normalized values')
    ax.legend()

    ax = axs_cadyn[1]
    ax.plot(t, catn_vec, label = 'f_stim = ' + str(params['f_stim']))
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

    # Print mean energetic rate over time course
    mean_energy_rates[idx] = np.mean(q_a + q_m)
    dt = np.diff(t, prepend=t[0])  # Compute time differences, prepend the first value to match dimensions
    dt[dt == 0] = 1e-10           # Replace zeros in time differences with a small epsilon
    # mean_force[idx] = np.mean(catn_vec)
    mean_force[idx] = np.sum(catn_vec * dt)

    # Print output
    print(f'f_stim = {params["f_stim"]} Hz, N_stim = {params["N_stim"]}')
    print(f'    Mean force over cycle  = {np.mean(catn_vec)} (Normalized)') 
    print(f'    Mean energetic rate over cycle  = {np.mean(q_a + q_m)} 1/s')

    #Plot output
    # fig, ax = plt.subplots(layout = 'constrained')
    # ax.plot(t, q_a, label = 'Activation heat') 
    # ax.plot(t, q_m, label = 'Maintenance heat')
    # ax.legend()
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Energetic rate (1/s)')

    # Plot the total energetic rate for each condition
    ax_energy_rate = axs_energy[0]
    ax_energy_rate.plot(t, q_a + q_m, label = 'f_stim = ' + str(params['f_stim']))

    # Plot the heat from the contraction 
    ax_energy_int = axs_energy[1]
    dt = np.diff(t, prepend=t[0])  # Compute time differences, prepend the first value to match dimensions
    dt[dt == 0] = 1e-10           # Replace zeros in time differences with a small epsilon
    ax_energy_int.plot(t, np.cumsum(q_a + q_m) * dt, label = 'f_stim = ' + str(params['f_stim']))

# Plot the mean force
ax_mech = axs_cadyn[2]
ax_mech.plot(f_stim, mean_force)
ax_mech.set_xlabel('Stimulation frequency (Hz)')
ax_mech.set_ylabel('Mean force (Normalized)')

ax_energy_rate.legend()
ax_energy_rate.set_xlabel('Time (s)')
ax_energy_rate.set_ylabel('Energetic rate (1/s)')

ax_energy_int.legend()
ax_energy_int.set_xlabel('Time (s)')
ax_energy_int.set_ylabel('Energy (Normalized)')

# Plot the mean energetic rate for each condition
ax_energy = axs_energy[2]
ax_energy.plot(f_stim, mean_energy_rates)
ax_energy.set_xlabel('Stimulation frequency (Hz)')
ax_energy.set_ylabel('Mean energetic rate (1/s)')

fig_energy.savefig('./Figures/Energy.pdf')
fig_cadyn.savefig('./Figures/CaMech.pdf')

plt.show()



