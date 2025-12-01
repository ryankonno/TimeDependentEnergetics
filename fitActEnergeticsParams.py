'''
Code to determine the activation and maintenance heat parameters for the model

'''

# Import 
import numpy as np 
from scipy.integrate import solve_ivp, cumtrapz
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt 
font = {'size'   : 14}
plt.rc('font', **font)
import matplotlib.cm as cmap
import pandas as pd
import itertools
import sys 
sys.path.append('./')


# Import the data to be used for comparison
data_exp = pd.read_csv('Data/Barclay1996_sol_heatrates_data.txt')




# Define the model parameters
params = {
    # Time parameters for setting up the protocol 
    't_start': 0, # s
    't_end': 2, # s
    'cycle_length': 1, # s, Defines the length of the cycle (used to set frequency of the contractions)
    'N_cycles': 10, # unitless, Number of cycles to simulate (rest period after N_cycles contractions)

    # General muscle parameters
    'rho0':  1e6,    # g/m^3, Density of muscle
    'max_iso_stress': 2.5e5, # N/m^2, Maximum isometric stress of the muscle

    'muscle': 'EDL', # Specify muscle parameters to be used in simulation

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
            'V_max_oxphos':  1.88, # mM/s, Vicini 2000... TBD
            # For Phillips Simulation 
            'atp_peak': 0.25, # 0.213, # mM/s Peak atp rate calculated based on initial heat rate and enthalpy of ATP from Phillips et al. 1993

            # May need to tune these parameters...
            'K_adp': 0.058, # mM, Vicini 2000.... TBD (may need to optimise for this parameter)
            # 'nh': 2.57, # unitless, VIcini 2000, .... TBD (may need to optimise for this parameter)
            'nh': 0.6456, # unitless, Varied

            # Values from Barclay and Weber 2004
            'F_0': 0, # N, 
            'l_0': 11e-3, # m, 
            'mass': 4.1e-3, # g, 
           
            # 'r_rec': 1 / 0.8 * 60e3, # J / mol, Assumes mitochondrial efficiency based on average mouse sol and edl
            'r_rec': 149835.87, # J / mol, Optimised value
            
            # Heat data used for optimisation 
            'heat_exp_rec': np.array((5.339988e-03, 5.726164e-02, 7.235249e-02, 7.009909e-02)), # J/F0l0, Slow, recovery heat 

            # Barclay and Weber 2004 experimental setup parameters 
            'velo_short': 1.3, # l0/s, Barclay and Weber 2004
            'freq': 250, # Hz, Frequency of stimulation 
            'max_dl': 0.1, # mm, Maximum length change

            # Activation model parameters 
            "Tau_1": 0.3, # Assume constant value from MCL (2023)
            # "Tau_2": 0.256, # Scaling based on MCL (2023)
            "Tau_2": 0.02,
            "K": 0.1025,
            "n": 4, # Hill coefficient for act mdoel

            
            # Mechanical parameters 
            'dedt_ce_max': 5, 
            'kappa': 0.18,

            # Initial energetics model 
            'r_am': 0.6177, # W/F_0/l_0, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_sl': 0.234, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

            # Heat rate parameters Barclay 1996
            'r_am_b1996': 0.6177, # W/F_0/l_0, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_sl_b1996': 0.234, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

        }, 
        'EDL': { 
            'c_c_tot': 29.5, # mM, Kushmerick et al. 1992 
            'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992 
            'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992 
            'Pi_0': 0, # mM,  Kushmerick et al. 1992 
            'V_max_oxphos': 1.88/2, # mM/s, Vicini 2000... TBD
            # For Phillips Simulation 
            'atp_peak': 0.25,# 0.213, # mM/s Peak atp rate calculated based on initial heat rate and enthalpy of ATP from Phillips et al. 1993

            # May need to tune these parameters...
            'K_adp': 0.058, # mM, Vicini 2000.... TBD (may need to optimise for this parameter)
            # 'nh': 2.57, # unitless, VIcini 2000, .... TBD (may need to optimise for this parameter)
            'nh': 1.180, # unitless, Varied

            'F_0': 0, # N, 
            'l_0': 8.9e-3, # m,
            'mass': 3.9e-3, # g, 
            
            # 'r_rec': 1 / 0.8 * 60e3, # J / mol, Assumes mitochondrial efficiency based on average mouse sol and edl
            'r_rec': 66536.221, # J / mol, Optimised value
            

            # Heat data used for optimisation 
            'heat_exp_rec': np.array((2.518854e-02, 6.404227e-02, 6.903081e-02, 7.313249e-02)), # J/F0l0, Fast, recovery heat 

            # Barclay and Weber 2004 experimental setup parameters 
            'velo_short': 2.8, # l0/s, Barclay and Weber 2004
            'freq': 125, # Hz, Frequency of stimulation 
            'max_dl': 0.2, # mm, Maximum length change


            # Activation model parameters 
            "Tau_1": 0.0422, # Very little change between fibre type - assume constant (BH, 2003)
            # "Tau_2": 0.256/2, # Decay constant for fast twitch Fibres assuming 1/2 rate (Baylor and Hollingworth, 2003)
            "Tau_2": 0.1/2, # Decay constant for fast twitch Fibres assuming 1/2 rate (Baylor and Hollingworth, 2003)
            "K": 0.1025,
            "n": 3, # Hill coefficient for activation model

            # Mechanical parameters 
            'dedt_ce_max': 10, 
            'kappa': 0.29,

            # Energetics model 
            'r_am': 2.792, # W/F_0/l_0, Maximum heat rate of isometric contraction (fast-type fibre)
            'r_sl': 0.697, # W/F_0/l_0, Maximum shortening heat rate (fast-type fibre)

             # Heat parameters from Barclay 1996
            'r_am_b1996': 2.792, # W/F_0/l_0, Maximum heat rate of isometric contraction (fast-type fibre)
            'r_sl_b1996': 0.697, # W/F_0/l_0, Maximum shortening heat rate (fast-type fibre)

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



        # Assume constant across all species and muscle fibre-types
        'V_ck_f': 100,# 100, # mM/s, Kushmerick 1998
        'K_b': 1.11, #mM, MacFarland 1994
        'K_ia': 0.135, # mM, MacFarland 1994
        'K_eq': 1.77e2, # ?, Assuming a pH of 7, Lawson 1979
        'K_iq': 3.5, # mM, MacFarland 1994
        'K_ib': 3.9, # mM, MacFarland 1994
        'K_p': 3.8, # mM, MacFarland 1994

        # Energetic constant to predict energetic rates 
        # 'r_rec': 1 / 0.8 * 60e3 # J / mol, Assumes mitochondrial efficiency based on average mouse sol and edl
        'Gatp': 60e3, # J/mol, Free energy of ATP (Barclay 2019)

        # Mechanical parameters (may not be used)
        'k_see': 0, # Unused



    
}




#####
# Load the models
from Models.MUActivationModel import ActivationModel
from Models.MechanicsModelSimple import MechModel 
from Models.MUEnergeticsModelSimple import EnergeticsModel

########
# Simulate the Ca dynamics and mechanics
# compute the ca dynamics for all contractions (same across conditions)
t_vec = np.linspace(params['t_start'], params['t_end'], int(1000 * params['t_end'])) 
act_model = ActivationModel(params[params['muscle']], t_vec, True)
# Compute the stimulus times 
stim_times = np.zeros_like(t_vec, dtype=int)
# determine stimulation frequency from params for the active muscle (fallback to 0)
period = 1.0 / params[params['muscle']]['freq']
# Get the firing times (accumulate into a Python list then convert)
t_fire_vec = []
if t_vec.size > 0:
    t_fire_prev = t_vec[0]
    t_fire_vec.append(t_fire_prev)
    # iterate remaining times and add a spike when period elapsed
    for t_val in t_vec[1:]:
        if t_val >= t_fire_prev + period - 1e-12:
            t_fire_vec.append(t_val)
            t_fire_prev = t_val
t_fire_vec = np.array(t_fire_vec)
# Map each spike time to the nearest index in t and mark stim_times
stim_times = np.zeros_like(t_vec, dtype=int)
if t_fire_vec.size > 0:
    t_arr = np.asarray(t_vec)
    for st in t_fire_vec:
        idx = int(np.argmin(np.abs(t_arr - st)))
        stim_times[idx] = 1
# Solve the activation model with the computed stim values 
idx_stims = np.nonzero(stim_times)[0]
stim_vec, ca_vec, catn_vec = act_model.runExcAct(idx_stims)
# Plot results to verify
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, ca_vec, label = '$Ca^{2+}$') 
ax.plot(t_vec, catn_vec, label = '$CaTn$') 
ax.legend()
ax.set_xlabel('Time ($s$)')
ax.set_ylabel('Normalised concentration')

########
# Define the models
# TODO: implement optimisation here
muscle = params['muscle']
mech_model = MechModel(params[muscle]['l_0'], params[muscle]['dedt_ce_max'], params[muscle]['kappa'],params['k_see'])

# Define the energetics model
energy_model = EnergeticsModel()

#####
# perform the optimisation 

# Define the shortening values for the optimsisation
v_short_vals = -np.arange(0, 2.5, 0.5) # s^{-1}

# Compute the experimental energetics
coeff = (params[muscle]['r_sl_b1996'], params[muscle]['r_am_b1996'])
tot_heat_exp = np.polyval(coeff, -v_short_vals)

# Define the optimisation function 
def fun_opt(x): 

    # Update values for parameters 
    params[muscle]['r_am'] = x[0]
    params[muscle]['r_sl'] = x[1]

    # Loop over the shortening values 
    mean_heat_tot = np.empty_like(v_short_vals, dtype = float)
    for idx_v, v_short in enumerate(v_short_vals):
        # Compute the strain-rate vector 
        dedt_ce = np.zeros_like(t_vec) + v_short * (t_vec >=1) * (t_vec < 1.1) 
        e_ce = cumtrapz(dedt_ce, t_vec, initial = 1.05)

        # Compute the force 
        force = catn_vec * mech_model.F_va(dedt_ce) * mech_model.F_la(e_ce)

        # Compute the energetics 
        q_a, q_m, q_sl, w = energy_model.actEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force, mech_model)

        # Compute the mean energetic rates over the 0.05s period where muscle has reached steady state during shortening
        # ie. from 1.05 to 1.1s
        t_range = (t_vec >= 1.05) * (t_vec  < 1.1)
        mean_heat_tot[idx_v] = np.mean(q_a[t_range] + q_m[t_range] + q_sl[t_range])

    # Compute the error 
    error = np.linalg.norm(tot_heat_exp - mean_heat_tot)

    return error 

# Perform optimisation 
opt_res = minimize(fun_opt, x0 = (params[muscle]['r_am'], params[muscle]['r_sl']))

print(opt_res) 
params[muscle]['r_am'] = opt_res.x[0]
params[muscle]['r_sl'] = opt_res.x[1]   

print('Optimised parameters: ')
print(f'r_am = {params[muscle]["r_am"]}, r_sl = {params[muscle]["r_sl"]}')

#######
# Plot a comparison of the optimised energetic rates

mean_heat_tot = np.empty_like(v_short_vals, dtype = float)
fig, ax = plt.subplots(layout = 'constrained')
fig_energy, axs_energy = plt.subplots(len(v_short_vals), 1, layout = 'constrained', figsize = (5,20))
for idx_v, v_short in enumerate(v_short_vals):
    # Solve Mechanics
    # Compute the strain-rate vector 
    dedt_ce = np.zeros_like(t_vec) + v_short * (t_vec >=1) * (t_vec < 1.1) 
    e_ce = cumtrapz(dedt_ce, t_vec, initial = 1.05)

    # Compute the force 
    force = catn_vec * mech_model.F_va(dedt_ce) * mech_model.F_la(e_ce)

    # Plot for verification
    ax.plot(t_vec, force, label = '$v_{short}$ = ' + str(v_short))

    # Compute the energetics 
    q_a, q_m, q_sl, w = energy_model.actEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force, mech_model)

    # plot for verification
    ax_energy = axs_energy[idx_v]
    ax_energy.plot(t_vec, q_a, label = '$q_a$')
    ax_energy.plot(t_vec, q_m, label = '$q_m$')
    ax_energy.plot(t_vec, q_sl, label = '$q_{sl}$')
    ax_energy.plot(t_vec, w, label = '$w$')
    ax_energy.legend()
    ax_energy.set_xlabel('Time ($s$)')
    ax_energy.set_ylabel('Energy rate ($F_0\,l_0\, s^{-1}$)')

    # Compute the mean energetic rates over the 0.05s period where muscle has reached steady state during shortening
    # ie. from 1.05 to 1.1s
    t_range = (t_vec >= 1.05) * (t_vec  < 1.1)
    mean_heat_tot[idx_v] = np.mean(q_a[t_range] + q_m[t_range] + q_sl[t_range]) 
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(-v_short_vals, mean_heat_tot) 
ax.plot(-v_short_vals, tot_heat_exp) 
plt.show()


