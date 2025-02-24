'''
Code to compute the time dependent energetic cost associated with both the initial reactions 
and the recovery processes. 

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
    # 't_end': 100, # s
    'N_contractions':50,
    'N_t': 100000, # int, Number of time steps (activation model)

    'f_stim': 20, # Hz, Stimulation frequency
    'N_stim': 6, # unitless, Number of stimulations

    'N_MU_sim': 1, # int, Number of MUs to stimulate    

    # MU Size properties (can be changed, these are just approximate values)
    # NOTE: These were estimated based on model estimates given RT experimentally
    #       can be changed....
    'F_0': 2e5 * 0.02e-5, # N, sigma_0 * CSA 
    'l_0': 6.83e-2, # m, resting fascicle length
    'mass': 0.0286, # g, weight of on MU (based on fraction of total muscle CSA and whole muscle mass of 143.1g)

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
        # 'nu': 1.72,

        # 'H_PCr': 36e3, # J/mol, Enthalpy of pcr

        'r_am': 0.6177, # W/F_0/l_0, Maximum heat rate of isometric contraction

                # Define parameters for the ODE 
        # .... In this code they are used as initial values for the optimization
        'k_recpcr': 0.0214, # 1/s
        'k_m_1':  0.0367, # 1/s

        # Energetic constant to predict energetic rates
        # 'r_r': 522e-3, # J (umol)^-1, Calculated based on the glycogen and atp enthalpys
        'r_r': 0.04489659, # J (umol)^-1, Optimized to experimental data (Phillips et al. 1993)

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
N_contr_tot = 37 * N_contr_cycle # Number of contractions over a 5min period


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
# # Compute the stimulation frequency for each of the contractions 
# fig, ax = plt.subplots(layout = 'constrained')
# for i in range(np.size(rates)):
#     t_stim = t_stim_all[i][~np.isnan(t_stim_all[i])]
#     print(f'Mean stimulation frequency: {np.nanmean(1/np.diff(t_stim))} Hz')
#     ax.plot(t_stim[1:], 1/np.diff(t_stim), label = 'rate = ' + str(rates[i]))
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Firing rate (Hz)')
# ax.legend()
# plt.savefig('./Figures/ExperimentalFiringRates.pdf')

######
# Initialize figures for display 
fig_energy, axs_energy = plt.subplots(1, 2, figsize = (12,6), layout = 'constrained')
fig_cadyn, axs_cadyn = plt.subplots(1, 2, figsize = (12,6), layout = 'constrained')

import matplotlib.cm as cmap
c_map = cmap.viridis(np.linspace(0,1,np.size(rates))) # Add plus one to avoid yellow (hard to see)


######
# Output storage 
mean_energy_rates = np.zeros_like(rates, dtype=float)
mean_q_i = np.zeros_like(rates, dtype=float)
mean_q_r = np.zeros_like(rates, dtype=float)

mean_force = np.zeros_like(rates, dtype=float)
time_active = np.zeros_like(rates, dtype = float)


# Loop through and simulate the contraction
for idx, t_stim  in enumerate(t_stim_all): 
    #____________________________
    # Stimulation...

    # Define the time vector for this simulation 
    t_vec_single = t_mat_vec[idx]
  
    # Remove any nan values from the data 
    t_stim = t_stim[~np.isnan(t_stim)]

    # Find the indices in the time vector closest to stimulation times
    t_stim_idxs_single = np.searchsorted(t_vec_single, t_stim, side='left')

    # To simulate recovery heat rates, we need a longer repeated contractions 
    tend = N_contr_tot[idx] * t_vec_single[-1]
    N_t = len(t_vec_single) * N_contr_tot[idx]
    t_vec = np.linspace(params['t_start'], tend, N_t)

    # Get the indices for the full vector 
    t_stim_idxs = np.zeros(N_contr_tot[idx]* len(t_stim_idxs_single),dtype=int)
    for i in range(N_contr_tot[idx]):
        # print(i * len(t_vec_single))
        t_stim_idxs[i*len(t_stim_idxs_single):(i+1)*len(t_stim_idxs_single)] = t_stim_idxs_single + (i * len(t_vec_single))

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
    ax.plot(t_vec, ca_vec, label = 'rate = ' + str(rates[idx]), color = c_map[idx])
    ax.set_xlim((0,4))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('[Ca2+] (Normalized)')
    ax.legend()

    ax = axs_cadyn[1]
    ax.plot(t_vec, catn_vec, label = 'rate = ' + str(rates[idx]), color = c_map[idx])
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
    q_a, q_m = energy_model.actEnergetics(t_vec, ca_vec, catn_vec, params['Energetics_model']) # W/F_0/l_0

    # Convert from W/F_0/l_0 to W/g 
    s_f = params['F_0'] * params['l_0'] / params['mass']
    q_i_W_g = (q_a + q_m) * s_f

    # Compute the recovery energetic rate using teh bioenergetics model 
    from Models.BioenergeticsModel import Bioenergetics
    model = Bioenergetics(params['Energetics_model']['k_recpcr'], params['Energetics_model']['k_m_1'],t_vec, q_i_W_g)
    sol = model.solveODE(t_vec[0], t_vec[-1], t_vec) # Solve the IVP 
    PCr, activation = sol.y

    q_r_W_g = model.computeEnergetics(PCr, activation, params['Energetics_model']['r_r']) # W/g, Compute the recovery energetic rate

    # Scale from W/g to W
    q_r = q_r_W_g * params['mass']
    q_i = q_i_W_g * params['mass']

    # Plot PCr, activation levels, 
    # fig, ax = plt.subplots(layout = 'constrained')
    # ax.plot(t_vec, q_r) 
    # fig.show()

    # Print mean energetic rate over time course
    mean_energy_rates[idx] = np.mean(q_i + q_r)
    mean_q_i[idx] = np.mean(q_i)
    mean_q_r[idx] = np.mean(q_r)

    # Compute the mean rates from 3min to 4min period 
    #  coresponds to contr 184s to the end 
    #   so take mean from 184*fsamp:end
    mean_energy_rates[idx] = np.mean(q_i[184 * fsamp:] + q_r[184 * fsamp:])
    mean_q_i[idx] = np.mean(q_i[184 * fsamp:])
    mean_q_r[idx] = np.mean(q_r[184 * fsamp:])


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
    print(f'    Mean energetic rate over cycle  = {mean_energy_rates[idx]} W')


    #Plot output
    # fig, ax = plt.subplots(layout = 'constrained')
    # ax.plot(t, q_a, label = 'Activation heat') 
    # ax.plot(t, q_m, label = 'Maintenance heat')
    # ax.legend()
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Energetic rate (1/s)')

    # Plot the total energetic rate for each condition
    ax_energy_rate = axs_energy[0]
    # ax_energy_rate.set_xlim((0,4))
    # ax_energy_rate.plot(t_vec, q_i, label = 'q_i, rate = ' + str(rates[idx]), color = c_map[idx])
    ax_energy_rate.plot(t_vec, q_r, label = 'q_r, rate = ' + str(rates[idx]), color = c_map[idx], ls = ':')

    # Plot the heat from the contraction 
    ax_energy_int = axs_energy[1]
    # ax_energy_int.set_xlim((0,4))
    dt = np.diff(t_vec, prepend=t_vec[0])  # Compute time differences, prepend the first value to match dimensions
    dt[dt == 0] = 1e-10           # Replace zeros in time differences with a small epsilon
    ax_energy_int.plot(t_vec, np.cumsum(q_i) * dt, label = 'q_i, rate = ' + str(rates[idx]), color = c_map[idx])
    ax_energy_int.plot(t_vec, np.cumsum(q_r) * dt, label = 'q_r, rate = ' + str(rates[idx]), color = c_map[idx], ls = ':')



######
# Plot for the mean quantities
fig_means, axs_means = plt.subplots(1, 3, figsize = (12,4), layout = 'constrained')
# Plot the mean force
ax_mech = axs_means[0]
ax_mech.plot(rates, mean_force)
ax_mech.set_xlabel('Contraction rate (1/s)')
ax_mech.set_ylabel('Mean force (Normalized)')

ax_energy_rate.legend()
ax_energy_rate.set_xlabel('Time (s)')
ax_energy_rate.set_ylabel('Energetic rate (W)')

ax_energy_int.legend()
ax_energy_int.set_xlabel('Time (s)')
ax_energy_int.set_ylabel('Energy (J)')

# Plot the mean energetic rate for each condition
ax_energy = axs_means[1]
ax_energy.plot(rates, mean_energy_rates, label = 'q_i + q_r')
ax_energy.plot(rates, mean_q_i, label = 'q_i')
ax_energy.plot(rates, mean_q_r, label = 'q_r')
ax_energy.legend()
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



