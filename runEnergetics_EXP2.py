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

        # Energetics parameters 
    'Energetics_model':{
        # 'nu': 1.72,

        # 'H_PCr': 36e3, # J/mol, Enthalpy of pcr 

        'r_am': 0.6177, # W/F_0/l_0, Maximum heat rate of isometric contraction (slow-type fibre)
        'r_sl': 0.234, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)
        # 'r_am': 2.792, # W/F_0/l_0, Maximum heat rate of isometric contraction (fast-type fibre)
        # 'r_sl': 0.697, # W/F_0/l_0, Maximum shortening heat rate (fast-type fibre)

        # Energetic constant to predict energetic rates
        # 'r_r': 522e-3, # J (umol)^-1, Calculated based on the glycogen and atp enthalpys
        'r_r': 0.04489659, # J (umol)^-1, Optimized to experimental data (Phillips et al. 1993)
    },

    # Mouse data 
    'SOL': {
        # Slow data
        # 'c_c_tot': 25.9, # mM, Kushmerick et al. 1992 
        # 'c_atp_0': 3.3, # mM,  Kushmerick et al. 1992 
        # 'c_pcr_0': 11.4, # mM,  Kushmerick et al. 1992 
        # Fast data 
        'c_c_tot': 29.5, # mM, Kushmerick et al. 1992 
        'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992 
        'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992 
        # 'Pi_0': 6, # mM,  Kushmerick et al. 1992 
        'V_max_oxphos':  1.88, # mM/s, Opt to Phillips
        # 'V_max_oxphos':  5, # mM/s, Vicini 2000... TBD
        # For Phillips Simulation 
        'atp_peak': 0.25,# 0.213, # mM/s Peak atp rate calculated based on initial heat rate and enthalpy of ATP from Phillips et al. 1993

        'F_0': 0.0102, # N, 
        'l_0': 9.5e-3, # m,
        'mass': 1.99e-3, # g, 
        # Energetic constant to predict energetic rates 
        # 'r_rec': (1 - 0.76) / 0.76 * 32 * 60e3, # J/ s / L, Assumes mitochondrial efficiency based on average mouse sol 
        # 'r_rec': (1 - 0.76) * 2810e3, # J/ s / L, Assumes mitochondrial efficiency based on average mouse sol 
        'r_rec': 0.04489659, # J (umol)^-1, Optimized to expe rimental data (Phillips et al. 1993)

        'duty_cycle':  0.16, # unitless, Duty cycle of initial heat rate

    }, 
    'EDL': { 
        'c_c_tot': 29.5, # mM, Kushmerick et al. 1992 
        'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992 
        'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992 
        'Pi_0': 0, # mM,  Kushmerick et al. 1992 
        'V_max_oxphos': 1.88/2, # mM/s, Opt to Phillips
        # 'V_max_oxphos':  2.5, # mM/s, Vicini 2000... TBD
        # For Phillips Simulation 
        'atp_peak': 0.25,# 0.213, # mM/s Peak atp rate calculated based on initial heat rate and enthalpy of ATP from Phillips et al. 1993

        'F_0': 0.0102, # N, 
        'l_0': 10e-3, # m,
        'mass': 2.85e-3, # g, 

        # Energetic constant to predict energetic rates 
        # 'r_rec': (1 - 0.86) / 0.86 * 32 * 60e3, # J/ s / L, Assumes mitochondrial efficiency based on average mouse sol 
        # 'r_rec': (1 - 0.86) * 2810e3, # J/ s / L, Assumes mitochondrial efficiency based on average mouse sol 
        'r_rec': 0.04489659, # J (umol)^-1, Optimized to experimental data (Phillips et al. 1993)

        'duty_cycle':  0.04, # unitless, Duty cycle of initial heat rate
    },
    
    # 'c_c_tot': 42, # mM, Harris 1974
    # 'c_atp_0': 8.2, # mM, Harris 1974
    # 'c_pcr_0': 32, # mM, Approximate value Vicini 2000
    # # 'c_c_tot': 20, # mM, Grassi 1998
    # # 'c_atp_0': 6.5, # mM, Grassi 1998
    # 'Pi_0': 3.183, # mM, Vicini 2000

    # 'V_max_oxphos': 0.5, # mM/s, Vicini 2000... TBD (may need to optimise for this parameter)
    # 'V_max_oxphos': 14.8 / 60, # mM/s, Vicini 2000... TBD (may need to optimise for this parameter)

    # Contraction dependent
    # NOTE: k_rest is not used in this implementation
    'k_rest': 0,# 0.0014, # 1/s, Vicini 2000, estimated off of experimental data Blei et al. 1993
    'k_stim': 0.0139,  # 1/s, Vicini 2000, estimated from exp data Blei et al. 1993
    'k_post': 0.9 * 0.0139,  # 1/s, Vicini 2000, estimated from exp data Blei et al. 1993 NOTE: cannot find value for this rate... assume half of stim?

    # May need to tune these parameters...
    'K_adp': 0.058, # mM, Vicini 2000.... TBD (may need to optimise for this parameter)
    'nh': 2.57, # unitless, VIcini 2000, .... TBD (may need to optimise for this parameter)

    # Assume constant across all species and muscle fibre-types
    'V_ck_f': 1000,# 100, # mM/s, Kushmerick 1998
    'K_b': 1.11, #mM, MacFarland 1994
    'K_ia': 0.135, # mM, MacFarland 1994
    'K_eq': 1.77e2, # ?, Assuming a pH of 7, Lawson 1979
    'K_iq': 3.5, # mM, MacFarland 1994
    'K_ib': 3.9, # mM, MacFarland 1994
    'K_p': 3.8, # mM, MacFarland 1994
}

###
# Set the MU sizes (CSA)
scales = np.array((0.9,0.925, 0.95, 0.975, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 4)) 
# scales = np.linspace(0.9, 4, 50)
mu_csa_vec = scales * params['mu_csa']
mu_mass_vec = scales * params['mass']
# mu_csa_vec =  np.array((params['mu_csa'], 2 * params['mu_csa'], 3 * params['mu_csa'], 4 * params['mu_csa']))
# mu_mass_vec = np.array((params['mass'], 2 * params['mass'], 3 * params['mass'], 4 * params['mass']))
F_0_vec = mu_csa_vec * params['sigma_0']

force_opt = 0.035252 # N, Desired optimized force for each MU

# Initialize storage vectors 
force = np.empty_like(mu_csa_vec,dtype=object)
catn_vec = np.empty_like(mu_csa_vec,dtype=object)
ca_vec = np.empty_like(mu_csa_vec,dtype=object)
force_mean = np.empty_like(mu_csa_vec,dtype=float)
r_stim_vec = np.empty_like(mu_csa_vec,dtype=float)
t_fire_idxs = np.empty_like(mu_csa_vec, dtype = object)

q_tot_vec = np.empty_like(mu_csa_vec, dtype = object)
q_i_vec = np.empty_like(mu_csa_vec, dtype = object)
q_r_vec = np.empty_like(mu_csa_vec, dtype = object)

q_tot_mean = np.empty_like(mu_csa_vec, dtype = float)
q_i_mean = np.empty_like(mu_csa_vec, dtype = float)
q_r_mean = np.empty_like(mu_csa_vec, dtype = float)


###
def computeMUStim(r_stim, t_vec_): 
    # Now get the firing times and then the indices in t_vec
    period = 1/np.maximum(1,r_stim)   
    N_t_fire = params['t_end'] / period
    t_fire_ideal = [i * period for i in range(int(N_t_fire))] # Ideal firing times 
    t_fire_idxs_ = np.searchsorted(t_vec_, t_fire_ideal, side='left') # Get the indices for the first MU
    return t_fire_idxs_

# Set the time vector 
t_vec = np.linspace(params['t_start'], params['t_end'], params['t_end'] * params['fsamp'])


from Models.MUActivationModel import ActivationModel
def computeActivation(t_vec_, t_fire_idxs_): 
    # Initialize 
    act_model = ActivationModel(params['Activation_model'], t_vec_, True)

    # Run the Ca dynamics
    stim_vec, ca_vec, catn_vec = act_model.runExcAct(t_fire_idxs_)

    return ca_vec, catn_vec


### 
# Compute the energetics 
from Models.MUEnergeticsModel import EnergeticsModel
def computeEnergetics(t_vec_, ca_vec_, catn_vec_, F_0_, l_0_, mass_): 

    # Initialize energetics model 
    energy_model = EnergeticsModel()

    # Compute the energetics based on Ca and CaTn 
    q_a, q_m = energy_model.actEnergetics(t_vec_, ca_vec_, catn_vec_, params['Energetics_model']) # W/F_0/l_0

    # Convert from W/F_0/l_0 to W/g 
    s_f = F_0_ * l_0_ / mass_
    q_i_W_g = (q_a + q_m) * s_f

    # Compute the recovery energetic rate using teh bioenergetics model 
    from Models.BioenergeticsModelKushmer import Bioenergetics
    model = Bioenergetics(params, t_vec_, q_i_W_g, muscle = 'SOL')
    t_span = (t_vec_[0], t_vec_[-1])
    sol = model.solveBioenergetics(t_span, params['SOL']['c_atp_0']) # Solve the IVP 
    c_atp, c_pcr = sol.y

    q_r_W_g = model.computeRecoveryEnergetics(c_atp) # W/g, Compute the recovery energetic rate

    # Scale from W/g to W
    q_r = q_r_W_g * mass_
    q_i = q_i_W_g * mass_

    # Compute the total heat rate
    q_tot = q_r + q_i 

    return q_tot, q_i, q_r


### 
# Compute the force trace for the smallest MU and compute the mean force
#  Use if we to compute the optimized force based on one MU

# Define the set firing times for the smaller MU
# r_stim_vec[0] = params['r_stim_1'] 
# print(f'Firing rate for MU0: {r_stim_vec[0]}')
# t_fire_idxs[0] = computeMUStim(r_stim_vec[0], t_vec)

# ca_vec[0], catn_vec[0] = computeActivation(t_vec, t_fire_idxs[0])

# # Compute the force based on the CSA 
# force[0] = catn_vec[0] * mu_csa_vec[0] * params['sigma_0']

# # Save the mean force 
# force_mean[0] = np.mean(force[0])

# q_tot_vec[0], q_i_vec[0], q_r_vec[0] = computeEnergetics(t_vec, ca_vec[0], catn_vec[0], F_0_vec[0], params['l_0'], mu_mass_vec[0])
# q_tot_mean[0], q_i_mean[0], q_r_mean[0] = np.mean(q_tot_vec[0]), np.mean(q_i_vec[0]), np.mean(q_r_vec[0])

# # Plot Ca dynamics for the first MU
# fig_cadyn, axs_cadyn = plt.subplots(1, 2, figsize = (12,6), layout = 'constrained')

# ax = axs_cadyn[0]
# ax.plot(t_vec, ca_vec[0] )
# ax.set_xlim((0,4))
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('[Ca2+] (Normalized)')

# ax = axs_cadyn[1]
# ax.plot(t_vec, force[0])
# ax.set_xlim((0,4))
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Force (N)')
# plt.show() 

### 
# Compute stimulation frequency required to match the mean force between the MUs
for mu, mu_csa in enumerate(mu_csa_vec):
    # Adjust mu number 
    # mu += 1

    print('__________________________________')
    print(f'Optimizing MU: {mu}')

    # Define the mass 
    mu_mass = mu_mass_vec[mu]

    # Define a function to optimize
    def f_opt(r_stim_mu):

        # Compute the time vector 
        t_fire_idxs[mu] = computeMUStim(r_stim_mu, t_vec)

        # Compute the activation level 
        _, catn_vec_ = computeActivation(t_vec, t_fire_idxs[mu])

        # Compute the force 
        force_mean[mu] = np.mean(catn_vec_ * mu_csa_vec[mu] * params['sigma_0'])

        # Error 
        error = np.abs(force_opt - force_mean[mu]) 

        return error 
    
    # Optimize over the stimulation frequency 
    from scipy.optimize import minimize_scalar
    opt_res = minimize_scalar(f_opt, bounds = (1,50))

    # Check if the optimization was successful
    if opt_res.success:
        # Save the recorded stimulation frequency
        r_stim_vec[mu] = opt_res.x
        print(f'    Optimized firing frequency: {r_stim_vec[mu]}')
        print(f'    Error: {opt_res.fun}')

        # Rerun and save the model results with the optimal stim frequency 
        t_fire_idxs[mu] = computeMUStim(r_stim_vec[mu], t_vec)
        ca_vec[mu], catn_vec[mu] = computeActivation(t_vec, t_fire_idxs[mu])
        force[mu] = catn_vec[mu] * mu_csa_vec[mu] * params['sigma_0']
        force_mean[mu] = np.mean(force[mu])
        print(f'    Mean force: {force_mean[mu]}')

        # Compute the energetics
        q_tot_vec[mu], q_i_vec[mu], q_r_vec[mu] = computeEnergetics(t_vec, ca_vec[mu], catn_vec[mu], F_0_vec[mu], params['l_0'], mu_mass_vec[mu])
        q_tot_mean[mu], q_i_mean[mu], q_r_mean[mu] = np.mean(q_tot_vec[mu]), np.mean(q_i_vec[mu]), np.mean(q_r_vec[mu])
    else:
        print(f'Optimization failed for MU: {mu}')



#### 
# Plotting 
import matplotlib.cm as cmap
c_map = cmap.viridis(np.linspace(0,1,np.size(mu_csa_vec)))

# Plot the corresponding results for all of the MUs
fig, axs = plt.subplots(1, 3, figsize = (12,4), layout = 'constrained') 
ax = axs[0] 
ax.set_xlabel('Time (s)')
ax.set_ylabel('q_{tot} (W)')
ax = axs[1] 
ax.set_xlabel('Time (s)')
ax.set_ylabel('q_{i} (W)')

ax = axs[2] 
ax.set_xlabel('Time (s)')
ax.set_ylabel('Energy (J)')

from scipy.integrate import cumulative_trapezoid
for mu in range(np.size(mu_csa_vec)):
    ax = axs[0] 
    ax.plot(t_vec, cumulative_trapezoid(q_tot_vec[mu],t_vec, initial=0), label  = None) 
    ax = axs[1] 
    ax.plot(t_vec, cumulative_trapezoid(q_i_vec[mu],t_vec, initial=0), label  = None) 
    ax = axs[2] 
    ax.plot(t_vec, cumulative_trapezoid(q_r_vec[mu],t_vec, initial=0), label  = None) 

# ax.legend()


# ax = axs[3] 
# ax.set_xlabel('Motor unit CSA (m^2)')
# ax.set_ylabel('Mean energetic rate (W)')
# ax.plot(mu_csa_vec, q_tot_mean, label = 'q_{tot}')
# ax.plot(mu_csa_vec, q_i_mean, label = 'q_i')
# ax.plot(mu_csa_vec, q_r_mean, label = 'q_r')
# ax.legend()

fig.savefig('./Figures/MUCompEnergy_detailed_EXP2.pdf')

# Plot the corresponding results for all of the MUs
fig, axs = plt.subplots(1, 2, figsize = (12,6), layout = 'constrained') 

ax = axs[0] 
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$q_{r}$ (W)')
for mu in range(np.size(mu_csa_vec)):
    ax = axs[0] 
    ax.plot(t_vec, q_r_vec[mu], label  = 'MU CSA ' + str(mu_csa_vec[mu]) + r' $m^{2}$', color = c_map[mu]) 
ax.legend()

ax = axs[1]
ls_opts = (':','-')

for mu in range(np.size(mu_csa_vec)):    
    ax.set_xlabel(f'Time')
    ax.set_ylabel(f'Cumulative mean energy rate (W)')
    if mu == 1: 
        ax.plot(t_vec, np.cumsum(q_tot_vec[mu]) / np.arange(1,np.size(q_tot_vec[mu]) + 1), label = r'$q_{tot}$, ' + f'MU size = {mu_csa_vec[mu]}', color = c_map[mu])
        ax.plot(t_vec, np.cumsum(q_i_vec[mu]) / np.arange(1,len(q_i_vec[mu]) + 1), label = r'$q_{i}$, ' + f'MU size = {mu_csa_vec[mu]}', ls = ':', color = c_map[mu])
        ax.plot(t_vec, np.cumsum(q_r_vec[mu]) / np.arange(1,len(q_r_vec[mu]) + 1), label = r'$q_{r}$, ' + f'MU size = {mu_csa_vec[mu]}', ls = '--', color = c_map[mu])  
    else: 
        ax.plot(t_vec, np.cumsum(q_tot_vec[mu]) / np.arange(1,np.size(q_tot_vec[mu]) + 1), color = c_map[mu], label = None)
        ax.plot(t_vec, np.cumsum(q_i_vec[mu]) / np.arange(1,len(q_i_vec[mu]) + 1), ls = ':', color = c_map[mu], label = None)
        ax.plot(t_vec, np.cumsum(q_r_vec[mu]) / np.arange(1,len(q_r_vec[mu]) + 1), ls = '--', color = c_map[mu], label = None)

# ax.legend(fontsize = 8)


# ax = axs[2] 
# ax.set_xlabel('Motor unit CSA (m^2)')
# ax.set_ylabel('Mean energetic rate (W)')
# ax.plot(mu_csa_vec, q_tot_mean, label = 'q_{tot}')
# ax.plot(mu_csa_vec, q_i_mean, label = 'q_i')
# ax.plot(mu_csa_vec, q_r_mean, label = 'q_r')
# ax.legend()

fig.savefig('./Figures/MUCompEnergy_EXP2.pdf')



# Plot the force results
fig, axs = plt.subplots(1, 2,figsize = (8,4)) 
ax = axs[0] 
ax.set_xlabel('Time (s)')
ax.set_ylabel('Firing rate (Hz)')
ax.plot(mu_csa_vec, r_stim_vec, label  = 'MU: ' + str(mu)) 

ax = axs[1] 
ax.set_xlabel('MU Size (m^2)')
ax.set_ylabel('Mean force (N)')
ax.plot(mu_csa_vec, force_mean)

fig.savefig('./Figures/MUCompForce_EXP2.pdf')

# Plot the surface plots for the cumulative energy rates
fig = plt.figure(figsize = (12,12),layout = 'constrained') 
plt.rcParams.update({'font.size': 12}) # Update font size

# Prepare data for the surface plot
X, Y = np.meshgrid(mu_csa_vec, t_vec)
energy_tot = np.zeros_like(X, dtype=float)
q_tot_mat = np.zeros_like(X, dtype=float)
q_i_mat = np.zeros_like(X, dtype=float)
q_r_mat = np.zeros_like(X, dtype=float)

for mu in range(np.size(mu_csa_vec)):
    energy_tot[:, mu] = cumulative_trapezoid(q_tot_vec[mu],t_vec, initial=0)
    
    q_tot_mat[:, mu] = np.cumsum(q_tot_vec[mu]) / np.arange(1, np.size(q_tot_vec[mu]) + 1)
    q_i_mat[:, mu] = np.cumsum(q_i_vec[mu]) / np.arange(1, np.size(q_i_vec[mu]) + 1)
    q_r_mat[:, mu] = np.cumsum(q_r_vec[mu]) / np.arange(1, np.size(q_r_vec[mu]) + 1)

padval = 5
ax = fig.add_subplot(221, projection='3d')
ax.set_ylabel('Time (s)', labelpad=padval)
ax.set_xlabel('Motor unit CSA (m^2)', labelpad=padval)
ax.set_zlabel('Total energy (J)', labelpad=padval)
surf = ax.plot_surface(X, Y, energy_tot, cmap='inferno')

ax = fig.add_subplot(222, projection='3d')
ax.set_ylabel('Time (s)', labelpad=padval)
ax.set_xlabel('Motor unit CSA (m^2)', labelpad=padval)
ax.set_zlabel('Cumulative mean energetic rate (J)', labelpad=padval)
surf = ax.plot_surface(X, Y, q_tot_mat, cmap='inferno')

ax = fig.add_subplot(223, projection='3d')
ax.set_ylabel('Time (s)', labelpad=padval)
ax.set_xlabel('Motor unit CSA (m^2)', labelpad=padval)
ax.set_zlabel('Cumulative mean intitial rate (J)', labelpad=padval)
surf = ax.plot_surface(X, Y, q_i_mat, cmap='inferno')

ax = fig.add_subplot(224, projection='3d')
ax.set_ylabel('Time (s)', labelpad=padval)
ax.set_xlabel('Motor unit CSA (m^2)', labelpad=padval)
ax.set_zlabel('Cumulative mean recovery rate (J)', labelpad=padval)
surf = ax.plot_surface(X, Y, q_r_mat, cmap='inferno')


fig.savefig('./Figures/MUCompEnergySurface_EXP2.jpg')
# fig.savefig('./Figures/MUCompEnergySurface_EXP2.pdf')

######
# Plot individual surface plots 

plt.rcParams.update({'font.size': 12})  # Update font size

# Prepare data for the surface plot
X, Y = np.meshgrid(mu_csa_vec * 1e6, t_vec)
energy_tot = np.zeros_like(X, dtype=float)
q_tot_mat = np.zeros_like(X, dtype=float)
q_i_mat = np.zeros_like(X, dtype=float)
q_r_mat = np.zeros_like(X, dtype=float)

for mu in range(np.size(mu_csa_vec)):
    energy_tot[:, mu] = cumulative_trapezoid(q_tot_vec[mu], t_vec, initial=0)
    q_tot_mat[:, mu] = np.cumsum(q_tot_vec[mu]) / np.arange(1, np.size(q_tot_vec[mu]) + 1)
    q_i_mat[:, mu] = np.cumsum(q_i_vec[mu]) / np.arange(1, np.size(q_i_vec[mu]) + 1)
    q_r_mat[:, mu] = np.cumsum(q_r_vec[mu]) / np.arange(1, np.size(q_r_vec[mu]) + 1)

padval = 7

# Plot the surface for total energy
fig1 = plt.figure(figsize=(5, 4))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.set_ylabel('Time (s)', labelpad=padval)
ax1.set_xlabel(r'Motor unit CSA ($mm^2$)', labelpad=padval)
ax1.set_zlabel('Total energy (J)', labelpad=padval)
surf1 = ax1.plot_surface(X, Y, energy_tot, cmap='inferno')
fig1.savefig('./Figures/MUCompEnergyTotal_EXP2.jpg')

# Plot the surface for cumulative mean energetic rate
fig2 = plt.figure(figsize=(5, 4))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_ylabel('Time (s)', labelpad=padval)
ax2.set_xlabel(r'Motor unit CSA ($mm^2$)', labelpad=padval)
ax2.set_zlabel('Mean energetic rate (J)', labelpad=padval)
surf2 = ax2.plot_surface(X, Y, q_tot_mat, cmap='inferno')
fig2.savefig('./Figures/MUCompEnergyRate_EXP2.jpg')

# Plot the surface for cumulative mean initial rate
fig3 = plt.figure(figsize=(5, 4))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.set_ylabel('Time (s)', labelpad=padval)
ax3.set_xlabel(r'Motor unit CSA ($mm^2$)', labelpad=padval)
ax3.set_zlabel('Mean initial rate (J)', labelpad=padval)
surf3 = ax3.plot_surface(X, Y, q_i_mat, cmap='inferno')
fig3.savefig('./Figures/MUCompInitialRate_EXP2.jpg')

# Plot the surface for cumulative mean recovery rate

fig4 = plt.figure(figsize=(5, 4))
ax4 = fig4.add_subplot(111, projection='3d')
ax4.set_ylabel('Time (s)', labelpad=padval)
ax4.set_xlabel(r'Motor unit CSA ($mm^2$)', labelpad=padval)
ax4.set_zlabel('Mean recovery rate (J)', labelpad=padval)
surf4 = ax4.plot_surface(X, Y, q_r_mat, cmap='inferno')

fig.tight_layout()
# fig.subplots_adjust(left=-1.5, right=1.5, top=0.5, bottom=-0.5) 

fig4.savefig('./Figures/MUCompRecoveryRate_EXP2.jpg')



plt.show()
