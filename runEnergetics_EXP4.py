'''
Code to compare the energetic cost of two motor units of different sizes with varying firing rates. This code runs the full energetic cost code.

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
    't_end': 100, # s
    'N_contractions':50,
    'fsamp': 2048, # Hz
    'N_t': 100000, # int, Number of time steps (activation model)

    'r_stim_1': 25, # Hz, Stimulation frequency for the smaller MU

    'N_MU_sim': 1, # int, Number of MUs to stimulate    

    # MU Size properties (can be changed, these are just approximate values)
    # NOTE: These were estimated based on model estimates given RT experimentally
    #       can be changed....
    'sigma_0': 2e5, # Pa, Maximum isometric stress
    'mu_csa': 0.02e-5, # m^2, motor unit CSA
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

    # Mechanical parameters
    'Mechanics_model': {
        'dedt_ce_max_slow': 5, 
        'dedt_ce_max_fast': 10, 
        'kappa_slow': 0.17, 
        'kappa_fast': 0.25, 
        
        'F_la_width': 0.3,   # Width of the force-length relationship
        # 'k_see':      0.4 * (26e-2 * 1.5 - 26e-2)/ ((1.5 * 26e-2 - 26e-2 * 0.94)) ,  # Series elastic stiffness
        'k_see':      20000/ 15.8e-4 / 2.5e5 * 6.83e-2,  # Series elastic stiffness (should give 0.94 strain??)
        # 'k_see':      10000/ 15.8e-4 / 2.5e5 * 6.83e-2,  # Series elastic stiffness (gives 0.88 strain)

        'dedt_ce_max': 5,    # 1/s, Maximum shortening rate
        'kappa':       0.17, # unitless, Curvature F-V relationship

        'ode_max_step': 1e-3, # Maximum time step size in mechanics ODE solver

        'l_0': 6.83e-2, # m, Optimal fibre length
        'l_m': 26e-2, # m, Resting muscle length, (Ward et al. 2009)
        'l_mtu': 26e-2 * 1.5, # m, Muscle resting length
    },

    # Energetics parameters 
    'Energetics_model':{
        # 'nu': 1.72,

        # 'H_PCr': 36e3, # J/mol, Enthalpy of pcr

        'r_am': 0.6177, # W/F_0/l_0, Maximum heat rate of isometric contraction (slow-type fibre)
        'r_sl': 0.234, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)
        # 'r_am': 2.792, # W/F_0/l_0, Maximum heat rate of isometric contraction (fast-type fibre)
        # 'r_sl': 0.697, # W/F_0/l_0, Maximum shortening heat rate (fast-type fibre)

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
# Import the modelling data 

# Define the contractions to loop over.
contr_idx_list = np.array((0, 1, 2, 3))
rates = np.array((0.125, 0.25, 0.5, 1.0)) # Contraction rates 


# Initial data storage for the mu pool data
data = np.empty((np.max(contr_idx_list) + 1,), dtype = object)

for contr_rate_idx in contr_idx_list: 
    peak_force = 0.4 # Normalized to MVC

    datapath = 'Data/' + str(rates[contr_rate_idx]) + 'mvcs_data.npy' 
    print('Importing data from ' + datapath)

    data[contr_rate_idx] = np.load(datapath, allow_pickle=True).item()
    # print(data[contr_rate_idx])
    print(f'End time: {data[contr_rate_idx]["t_vec"][-1]}')

###
# Define locations to store the data
q_tot_vec =np.empty((np.max(contr_idx_list) + 1,), dtype = object)
q_i_vec = np.empty((np.max(contr_idx_list) + 1,), dtype = object)
q_r_vec = np.empty((np.max(contr_idx_list) + 1,), dtype = object)

q_tot_cummean = np.empty((np.max(contr_idx_list) + 1,), dtype = object)
q_i_cummean = np.empty((np.max(contr_idx_list) + 1,), dtype = object)
q_r_cummean = np.empty((np.max(contr_idx_list) + 1,), dtype = object)

q_tot_muscle_vec =np.empty((np.max(contr_idx_list) + 1,), dtype = object)
q_i_muscle_vec = np.empty((np.max(contr_idx_list) + 1,), dtype = object)
q_r_muscle_vec = np.empty((np.max(contr_idx_list) + 1,), dtype = object)
q_tot_muscle_cummean = np.empty((np.max(contr_idx_list) + 1,), dtype = object)
q_i_muscle_cummean = np.empty((np.max(contr_idx_list) + 1,), dtype = object)
q_r_muscle_cummean = np.empty((np.max(contr_idx_list) + 1,), dtype = object)

force_vec = np.empty((np.max(contr_idx_list) + 1,), dtype = object)
force_muscle_vec = np.empty((np.max(contr_idx_list) + 1,), dtype = object)
force_norm_vec = np.empty((np.max(contr_idx_list) + 1,), dtype = object)

### 
# Function to compute the energetics 
# TODO: Update this function to handle different fibre-types...
def computeEnergetics(t_vec_, ca_vec_, catn_vec_, F_0_, l_0_, mass_, e_m_ = None, dedt_m_ = None, force_ = None): 
    
    # Import the model 
    from Models.MUEnergeticsModel import EnergeticsModel

    # Initialize energetics model 
    energy_model = EnergeticsModel()

    # Compute the energetics based on Ca and CaTn 
    q_a, q_m, q_sl, w = energy_model.actEnergetics(t_vec_, ca_vec_, catn_vec_, params['Energetics_model'], e_m = e_m_, dedt_m = dedt_m_, force = force_, params_mech = params['Mechanics_model']) # W / F_0 / l_0

    # plt.figure() 
    # plt.plot(t_vec_, q_a)
    # plt.plot(t_vec_, q_m)
    # plt.plot(t_vec_, q_sl)
    # # plt.plot(t_vec_, w)
    # # plt.plot(sol.t[0:-100],PCr[0:-100]) 
    # plt.show()

    # Convert from W/F_0/l_0 to W/g 
    s_f = F_0_ * l_0_ / mass_
    q_i_W_g = (q_a + q_m + q_sl + w) * s_f

    # Compute the recovery energetic rate using teh bioenergetics model 
    from Models.BioenergeticsModel import Bioenergetics
    model = Bioenergetics(params['Energetics_model']['k_recpcr'], params['Energetics_model']['k_m_1'],t_vec_, q_i_W_g)
    sol = model.solveODE(t_vec_[0], t_vec_[-1], t_vec_) # Solve the IVP 
    PCr, activation = sol.y

    q_r_W_g = model.computeEnergetics(PCr, activation, params['Energetics_model']['r_r']) # W/g, Compute the recovery energetic rate

    # Scale from W/g to W
    q_r = q_r_W_g * mass_
    q_i = q_i_W_g * mass_

    # Compute the total heat rate
    q_tot = q_r + q_i 

    return q_tot, q_i, q_r

### 
# Loop over contractions and compute the energetic cost 
for contr_idx in contr_idx_list: 

    print('_________________________________________')
    print(f'Running contraction {contr_idx}')
    print(f'Contraction rate = {rates[contr_idx]}')

    # Unpack the data from the contraction
    t_vec = data[contr_idx]['t_vec']
    dt_mat = data[contr_idx]['dt_mat']
    ca_mat = data[contr_idx]['ca_vals']
    catn_mat = data[contr_idx]['act_mat']


    force_ = data[contr_idx]['force_m']
    e_m_ = data[contr_idx]['e_m']
    dedt_m_ = data[contr_idx]['dedt_m']
    t_vec_mech = data[contr_idx]['t_vec_mech']

    # Interpolate the mechanical vectors to the same time points as the rest 
    # force = np.interp(t_vec, t_vec_mech, force_)
    e_m = np.interp(t_vec, t_vec_mech, e_m_)
    dedt_m = np.interp(t_vec, t_vec_mech, dedt_m_)

    # Compute the mass of each motor unit based on its cross-sectional area
    mu_mass_vec = data[contr_idx]['csa_frac'] * params['mass']

    # Compute the maximum isometric force of the motor units based on the csa of the MU 
    F_0_vec = data[contr_idx]['sigma_0_mu'] * data[contr_idx]['csa']

    # Initialize storage vectors 
    q_tot_vec[contr_idx] = np.empty_like(dt_mat, dtype = object)
    q_i_vec[contr_idx] = np.empty_like(dt_mat, dtype = object)
    q_r_vec[contr_idx] = np.empty_like(dt_mat, dtype = object)
    q_tot_cummean[contr_idx] = np.empty_like(dt_mat, dtype = object)
    q_i_cummean[contr_idx] = np.empty_like(dt_mat, dtype = object)
    q_r_cummean[contr_idx] = np.empty_like(dt_mat, dtype = object)
    force_vec[contr_idx] = np.empty_like(dt_mat, dtype = object)
    force_norm_vec[contr_idx] = np.empty_like(dt_mat, dtype = object)

    # Loop over motor units and compute the energetic rates 
    # TODO: add multithreading
    # TODO: Account for different fibre-types
    for mu, dts in enumerate(dt_mat):
    # for mu, dts in enumerate(range(342,360)):

        # print(f'     MU = {mu}')

        # Compute the force from the motor unit 
        # Need to import the mech model to do this 
        from Models.MechanicsModel import MechModel 
        mech_model = MechModel(params['Mechanics_model'])
        force_norm_vec[contr_idx][mu] = mech_model.computeForce(catn_mat[:,mu], e_m, dedt_m)
        force_vec[contr_idx][mu] = force_norm_vec[contr_idx][mu] * F_0_vec[mu]

        # Compute the initial energetic rates 
        # Compute the energetics
        q_tot_vec[contr_idx][mu], q_i_vec[contr_idx][mu], q_r_vec[contr_idx][mu] = \
                        computeEnergetics(t_vec, ca_mat[mu], catn_mat[:,mu], F_0_vec[mu], params['l_0'], mu_mass_vec[mu], e_m, dedt_m, force_norm_vec[contr_idx][mu])
        q_tot_cummean[contr_idx][mu] = np.cumsum(q_tot_vec[contr_idx][mu]) / np.arange(1,np.size(q_tot_vec[contr_idx][mu]) + 1)
        q_i_cummean[contr_idx][mu] = np.cumsum(q_i_vec[contr_idx][mu]) / np.arange(1,np.size(q_i_vec[contr_idx][mu]) + 1)
        q_r_cummean[contr_idx][mu] = np.cumsum(q_r_vec[contr_idx][mu]) / np.arange(1,np.size(q_r_vec[contr_idx][mu]) + 1)

       
        
    # Combine the individual MU energetic rates into a full muscle energetic rate 
    q_tot_muscle_vec[contr_idx] = np.sum(q_tot_vec[contr_idx])
    q_i_muscle_vec[contr_idx] = np.sum(q_i_vec[contr_idx])
    q_r_muscle_vec[contr_idx] = np.sum(q_r_vec[contr_idx])
    q_tot_muscle_cummean[contr_idx] = np.cumsum(q_tot_muscle_vec[contr_idx]) / np.arange(1,np.size(q_tot_muscle_vec[contr_idx]) + 1)
    q_i_muscle_cummean[contr_idx] = np.cumsum(q_i_muscle_vec[contr_idx]) / np.arange(1,np.size(q_i_muscle_vec[contr_idx]) + 1)
    q_r_muscle_cummean[contr_idx] = np.cumsum(q_r_muscle_vec[contr_idx]) / np.arange(1,np.size(q_r_muscle_vec[contr_idx]) + 1)

    force_muscle_vec[contr_idx] = np.sum(force_vec[contr_idx])

    # Print the main output
    print(f'Mean force = {np.mean(force_muscle_vec[contr_idx])} N')
    print(f'Mean energetic rate = {np.mean(q_tot_muscle_cummean[contr_idx][-1])} W')
    print(f'Mean initial energetic rate = {np.mean(q_i_muscle_cummean[contr_idx][-1])} W')
    print(f'Mean recovery energetic rate = {np.mean(q_r_muscle_cummean[contr_idx][-1])} W')
    

# Save the energetic rates 
energy_out  = {
    'q_tot_vec': q_tot_vec, 
    'q_i_vec': q_tot_vec, 
    'q_r_vec': q_tot_vec, 
    'force': force_muscle_vec,
}
np.save('./Results/EXP4_energy.npy', energy_out)
# np.savetxt('./Results/Energy_CumMean.txt',q_tot_muscle_cummean)

### 
# Plot the energetic rates
fig, axs = plt.subplots( figsize = (6,4), layout = 'constrained') 
import matplotlib.cm as cmap
c_map = cmap.viridis(np.linspace(0,1,np.max(contr_idx_list) + 1))
ax = axs
for contr_idx in contr_idx_list:   
    ax.set_xlabel(f'Time')
    ax.set_ylabel(f'Cumulative mean energy rate (W)')
    ax.plot(data[contr_idx]['t_vec'], q_tot_muscle_cummean[contr_idx], label = r'$q_{tot}$, ' + f'Rate = {rates[contr_idx]}', color = c_map[contr_idx])
    ax.plot(data[contr_idx]['t_vec'], q_i_muscle_cummean[contr_idx], label = r'$q_{i}$, ' + f'Rate = {rates[contr_idx]}', ls = ':', color = c_map[contr_idx])
    ax.plot(data[contr_idx]['t_vec'], q_r_muscle_cummean[contr_idx], label = r'$q_{r}$, ' + f'Rate = {rates[contr_idx]}', ls = '--', color = c_map[contr_idx])
ax.legend(fontsize = 8)
plt.savefig('./Figures/Energy_CumMean_EXP4.pdf')
plt.show()


# Plot the force produced by the muscle
fig, axs = plt.subplots( figsize = (6,4), layout = 'constrained') 
import matplotlib.cm as cmap
c_map = cmap.viridis(np.linspace(0,1,np.max(contr_idx_list) + 1))
ax = axs
for contr_idx in contr_idx_list:   
    ax.set_xlabel(f'Time')
    ax.set_ylabel(f'Force (N)')
    ax.plot(data[contr_idx]['t_vec'], force_muscle_vec[contr_idx], label = f'Rate = {rates[contr_idx]}', color = c_map[contr_idx])
ax.legend(fontsize = 8)
plt.savefig('./Figures/Force_CumMean_EXP4.pdf')
plt.show()



# # Plot the corresponding results for all of the MUs
# fig, axs = plt.subplots(1, 4, figsize = (12,4), layout = 'constrained') 
# ax = axs[0] 
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('q_{tot} (W)')
# ax = axs[1] 
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('q_{i} (W)')
# ax = axs[2] 
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('q_{r} (W)')

# for mu in range(np.size(mu_csa_vec)):
#     ax = axs[0] 
#     ax.plot(t_vec, q_tot_vec[mu], label  = 'MU: ' + str(mu)) 
#     ax = axs[1] 
#     ax.plot(t_vec, q_i_vec[mu], label  = 'MU: ' + str(mu)) 
#     ax = axs[2] 
#     ax.plot(t_vec, q_r_vec[mu], label  = 'MU: ' + str(mu)) 

# ax.legend()


# ax = axs[3] 
# ax.set_xlabel('Motor unit CSA (m^2)')
# ax.set_ylabel('Mean energetic rate (W)')
# ax.plot(mu_csa_vec, q_tot_mean, label = 'q_{tot}')
# ax.plot(mu_csa_vec, q_i_mean, label = 'q_i')
# ax.plot(mu_csa_vec, q_r_mean, label = 'q_r')
# ax.legend()

# fig.savefig('./Figures/MUCompEnergy_detailed_EXP2.pdf')

# # Plot the corresponding results for all of the MUs
# fig, axs = plt.subplots(1, 2, figsize = (12,6), layout = 'constrained') 
# import matplotlib.cm as cmap
# c_map = cmap.viridis(np.linspace(0,1,np.size(mu_csa_vec)))
# ax = axs[0] 
# ax.set_xlabel('Time (s)')
# ax.set_ylabel(r'$q_{r}$ (W)')
# for mu in range(np.size(mu_csa_vec)):
#     ax = axs[0] 
#     ax.plot(t_vec, q_r_vec[mu], label  = 'MU size ' + str(mu_csa_vec[mu]) + r' $m^{2}$', color = c_map[mu]) 
# ax.legend()

# ax = axs[1]
# ls_opts = (':','-')

# for mu in range(np.size(mu_csa_vec)):    
#     ax.set_xlabel(f'Time')
#     ax.set_ylabel(f'Cumulative mean energy rate (W)')
#     ax.plot(t_vec, np.cumsum(q_tot_vec[mu]) / np.arange(1,np.size(q_tot_vec[mu]) + 1), label = r'$q_{tot}$, ' + f'MU size = {mu_csa_vec[mu]}', color = c_map[mu])
#     ax.plot(t_vec, np.cumsum(q_i_vec[mu]) / np.arange(1,len(q_i_vec[mu]) + 1), label = r'$q_{i}$, ' + f'MU size = {mu_csa_vec[mu]}', ls = ':', color = c_map[mu])
#     ax.plot(t_vec, np.cumsum(q_r_vec[mu]) / np.arange(1,len(q_r_vec[mu]) + 1), label = r'$q_{r}$, ' + f'MU size = {mu_csa_vec[mu]}', ls = '--', color = c_map[mu])
# ax.legend(fontsize = 8)


# # ax = axs[2] 
# # ax.set_xlabel('Motor unit CSA (m^2)')
# # ax.set_ylabel('Mean energetic rate (W)')
# # ax.plot(mu_csa_vec, q_tot_mean, label = 'q_{tot}')
# # ax.plot(mu_csa_vec, q_i_mean, label = 'q_i')
# # ax.plot(mu_csa_vec, q_r_mean, label = 'q_r')
# # ax.legend()

# fig.savefig('./Figures/MUCompEnergy_EXP2.pdf')



# # Plot the force results
# fig, axs = plt.subplots(1, 2,figsize = (8,4), layout = 'constrained') 
# ax = axs[0] 
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Force (N)')
# for mu in range(np.size(mu_csa_vec)):
#     ax.plot(t_vec, force[mu], label  = 'MU: ' + str(mu)) 

# ax = axs[1] 
# ax.set_xlabel('MU Size (m^2)')
# ax.set_ylabel('Mean force (N)')
# ax.plot(mu_csa_vec, force_mean)

# fig.savefig('./Figures/MUCompForce_EXP2.pdf')

# plt.show()
