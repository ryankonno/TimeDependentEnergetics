'''
Code to run single MU energetics with experimental data for MU firing rates 

Ryan Konno
School of Biomedical Sciences 
The University of Queensland
'''

# Import 
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 14
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

###
# Import the experimental data 
data = np.genfromtxt("Data/P11_Ramps_SingleMUTracked.csv", delimiter=",", dtype=float, encoding="utf-8").T

# Define the rates that are in the above data 
rates = np.array((1,0.5, 0.25, 0.125))
N_contr_cycle = np.array((8,4,2,1))
fsamp = 2048
t_init = np.array((0.1, 0.2, 0.3, 0.4)) # Time before the start of the contraction

# Remove 0 values 
data[data == 0] = np.nan
idx_stims_all = np.empty((np.size(rates)), dtype = object)
for i in range(np.size(rates)):
    idx_stims_all[i] = data[i,~np.isnan(data[i,:])]

######
# Crop the MU data to one contraction 
idx_cycles = fsamp * 1 /  rates * 0.4 * 2.5  # Define the number of points per cycle (fsamp*t_ramp*f_max*cycle)
idx_rec = np.array([arr[0] if arr.size > 0 else np.nan for arr in idx_stims_all]) # Get the first recruited data point 
idx_start_cycle = np.zeros_like(rates) # Initialize 
for i in range(len(rates)):
    idx_start_cycle[i] = np.floor(idx_rec[i] - t_init[i] * fsamp) # Start the cycle time from t_init before the first MU firing 
idx_end_cycle = np.floor(idx_start_cycle + idx_cycles) # 4x1, Get the last index of the cycle 

# Loop through and extract the contractions
data_crop = np.empty((np.size(rates)), dtype = object)
for i in range(np.size(rates)): 
    data_crop[i] = idx_stims_all[i][idx_stims_all[i] < idx_end_cycle[i]]  # Crop the data to only include one contraction 
idx_stim_all = data_crop - idx_start_cycle # Re order indices 

# Convert data from indices to time values 
t_stim_all = idx_stim_all / fsamp

# Define the time values 
t_end_vec = np.array((idx_cycles / fsamp), dtype = float)
t_mat_vec = np.empty((np.size(rates)), dtype = object)
for i in range(np.size(rates)): 
    t_mat_vec[i] = np.linspace(params['t_start'], t_end_vec[i], int(t_end_vec[i] * fsamp))

######
# Compute the stimulation frequency for each of the contractions 
fig, ax = plt.subplots(layout = 'constrained')
for i in range(np.size(rates)):
    t_stim = t_stim_all[i][~np.isnan(t_stim_all[i])]
    print(f'Mean stimulation frequency: {np.nanmean(1/np.diff(t_stim))} Hz')
    ax.plot(t_stim[1:], 1/np.diff(t_stim), label = 'rate = ' + str(rates[i]))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Firing rate (Hz)')
ax.legend()
plt.savefig('./Figures/ExperimentalFiringRates.pdf')

######
# Initialize figures for display 
fig_energy, axs_energy = plt.subplots(1, 2, figsize = (12,6), layout = 'constrained')
fig_cadyn, axs_cadyn = plt.subplots(1, 2, figsize = (12,6), layout = 'constrained')
fig_means, axs_means = plt.subplots(1, 3, figsize = (12,4), layout = 'constrained')

######
# Output storage 
mean_energy_rates = np.zeros_like(rates, dtype=float)
mean_force = np.zeros_like(rates, dtype=float)
time_active = np.zeros_like(rates, dtype = float)


# Loop through and simulate the contraction
for idx, t_stim  in enumerate(t_stim_all): 
    #____________________________
    # Stimulation...

    # Define the time vector for this simulation 
    t_vec = t_mat_vec[idx]
  
    # Remove any nan values from the data 
    t_stim = t_stim[~np.isnan(t_stim)]

    # Find the indices in the time vector closest to stimulation times
    t_stim_idxs = np.searchsorted(t_vec, t_stim, side='left')

    ######################################################
    '''
    Run the activation model 
    '''

    #  Import the activation model 
    from Models.MUActivationModel import ActivationModel

    # Initialize 
    act_model = ActivationModel(params['Activation_model'], t_vec, True)

    # Run the Ca dynamics
    stim_vec, ca_vec, catn_vec = act_model.runExcAct(t_stim_idxs)

    # Plot Ca dynamics
    ax = axs_cadyn[0]
    ax.plot(t_vec, ca_vec, label = 'rate = ' + str(rates[idx]))
    ax.set_xlim((0,4))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('[Ca2+] (Normalized)')
    ax.legend()

    ax = axs_cadyn[1]
    ax.plot(t_vec, catn_vec, label = 'rate = ' + str(rates[idx]))
    ax.set_xlim((0,4))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('[CaTn] Normalized')
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
    q_a, q_m = energy_model.actEnergetics(t_vec, ca_vec, catn_vec, params['Energetics_model'])

    # Print mean energetic rate over time course
    mean_energy_rates[idx] = np.mean(q_a + q_m)
    dt = np.diff(t_vec, prepend=t_vec[0])  # Compute time differences, prepend the first value to match dimensions
    dt[dt == 0] = 1e-10           # Replace zeros in time differences with a small epsilon
# 
    # mean_force[idx] = np.sum(catn_vec * dt)
    mean_force[idx] = np.mean(catn_vec)

    # Calculate time active 
    time_active[idx] = ((t_stim_idxs[-1] - t_stim_idxs[0]) / fsamp)  # s

    # Print output (NOTE: Indented values are calculated using simulation values)
    print('___________________________________________')
    print(f'Contraction = {rates[idx]} s^-1')
    print(f'    Total Time of simulation = {max(t_vec)} (s)')
    print(f'    Time from first stim to last stim = {time_active[idx]} (s)')
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
    ax_energy_rate.set_xlim((0,4))
    ax_energy_rate.plot(t_vec, q_a + q_m, label = 'rate = ' + str(rates[idx]))

    # Plot the heat from the contraction 
    ax_energy_int = axs_energy[1]
    ax_energy_int.set_xlim((0,4))
    dt = np.diff(t_vec, prepend=t_vec[0])  # Compute time differences, prepend the first value to match dimensions
    dt[dt == 0] = 1e-10           # Replace zeros in time differences with a small epsilon
    ax_energy_int.plot(t_vec, np.cumsum(q_a + q_m) * dt, label = 'rate = ' + str(rates[idx]))

# Plot the mean force
ax_mech = axs_means[0]
ax_mech.plot(rates, mean_force)
ax_mech.set_xlabel('Contraction rate (1/s)')
ax_mech.set_ylabel('Mean force (Normalized)')

ax_energy_rate.legend()
ax_energy_rate.set_xlabel('Time (s)')
ax_energy_rate.set_ylabel('Energetic rate (1/s)')

ax_energy_int.legend()
ax_energy_int.set_xlabel('Time (s)')
ax_energy_int.set_ylabel('Energy (Normalized)')

# Plot the mean energetic rate for each condition
ax_energy = axs_means[1]
ax_energy.plot(rates, mean_energy_rates)
ax_energy.set_xlabel('Contraction rate (1/s)')
ax_energy.set_ylabel('Mean energetic rate (1/s)')

# Plot the ratio of time active to actual time active 
ax_time = axs_means[2] 
ax_time.plot(rates, time_active/ (1/rates * 0.4))
ax_time.set_xlabel('Contraction rate (1/s)')
ax_time.set_ylabel('MU time active (computed/ideal)')

fig_energy.savefig('./Figures/Energy.pdf')
fig_cadyn.savefig('./Figures/CaMech.pdf')
fig_means.savefig('./Figures/MeanRates.pdf')
plt.show()



