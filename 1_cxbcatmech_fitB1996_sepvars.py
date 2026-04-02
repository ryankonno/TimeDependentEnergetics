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


# Data files for each muscle type (SOL = soleus/slow, EDL = fast)
data_files = {
    'SOL': 'Data/Barclay1996_sol_heatrates_data.txt',
    'EDL': 'Data/Barclay1996_edl_heatrates_data.txt',
}




# Define the model parameters
params = {
    # Time parameters for setting up the protocol 
    't_start': 0, # s
    't_end': 2, # s

    # General muscle parameters
    'rho0':  1e6,    # g/m^3, Density of muscle
    'max_iso_stress': 2.5e5, # N/m^2, Maximum isometric stress of the muscle

    'muscle': 'SOL', # Specify muscle parameters to be used in simulation

        # Mouse data 
        'SOL': {
            'freq': 75, # Hz, Frequency of stimulation 

            # Activation model parameters 
            "Tau_1": 0.0422, # Assume constant value from MCL (2023)
            # "Tau_2": 0.125, # Scaling based on MCL (2023)
            "Tau_2": 0.057, #  BH 2012            # "Tau_2": 0.02,
            "K": 0.1025,
            "n": 4, # Hill coefficient for act mdoel

            # Mechanical parameters 
            'dedt_ce_max': 5, 
            'kappa': 0.18,

            # Initial energetics model 
            # 'r_am': 0.6177, # W/F_0/l_0, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cxb': 0.6 * 0.6177, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cat': 0.4 * 0.6177, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_sl': 0.234, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

            # Heat rate parameters Barclay 1996
            'r_am_b1996': 0.6177, # W/F_0/l_0, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_sl_b1996': 0.234, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

        }, 
        'EDL': { 
            'freq': 125, # Hz, Frequency of stimulation 

            # Activation model parameters 
            "Tau_1": 0.0422, # Very little change between fibre type - assume constant (BH, 2003)
            # "Tau_2": 0.125/2, # Decay constant for fast twitch Fibres assuming 1/2 rate (Baylor and Hollingworth, 2003)
            "Tau_2": 0.011, # BH 2012
            "K": 0.1025,
            "n": 3, # Hill coefficient for activation model

            # Mechanical parameters 
            'dedt_ce_max': 10, 
            'kappa': 0.25,

            # Energetics model 
            # 'r_am': 2.792, # F0l0/s, Maximum heat rate of isometric contraction (fast-type fibre)
            'r_cxb': 0.6 * 2.792, # F0l0/s, Maximum heat rate of isometric contraction (fast-type fibre)
            'r_cat': 0.4 * 2.792, # F0l0/s, Maximum heat rate of isometric contraction (fast-type fibre)
            'r_sl': 0.697, # F0l0/s, Maximum shortening heat rate (fast-type fibre)

             # Heat parameters from Barclay 1996
            'r_am_b1996': 2.792, # F0l0/s, Maximum heat rate of isometric contraction (fast-type fibre)
            'r_sl_b1996': 0.697, # F0l0/s, Maximum shortening heat rate (fast-type fibre)

        },

        # Mechanical parameters (may not be used)
        'k_see': 0, # Unused
    
}

#####
# Load the models
from Models.MUActivationModel import ActivationModel
from Models.MechanicsModelSimple import MechModel 
from Models.MUEnergeticsModelSimple_SplitVars import EnergeticsModel


# Initialise plots for the model 
fig_hr, ax_hr = plt.subplots(layout='constrained')


########
# Loop over both muscle types: SOL (soleus/slow) and EDL (fast)
for muscle_type in ['SOL', 'EDL']:
    params['muscle'] = muscle_type
    muscle = muscle_type

    # Load the experimental data for this muscle
    data_exp = pd.read_csv(data_files[muscle_type])

    print(f'\n=== Processing {muscle_type} ({"slow/soleus" if muscle_type == "SOL" else "fast/EDL"}) ===')

    ########
    # Simulate the Ca dynamics and mechanics
    # compute the ca dynamics for all contractions (same across conditions)
    t_vec = np.linspace(params['t_start'], params['t_end'], int(1000 * params['t_end']))
    act_model = ActivationModel(params[params['muscle']], t_vec, True)
    # Compute the stimulus times
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
    # # Plot results to verify
    # fig, ax = plt.subplots(layout='constrained')
    # fig.suptitle(f'{muscle_type} - Activation')
    # ax.plot(t_vec, ca_vec, label='$Ca^{2+}$')
    # ax.plot(t_vec, catn_vec, label='$CaTn$')
    # ax.legend()
    # ax.set_xlabel('Time ($s$)')
    # ax.set_ylabel('Normalised concentration')

    ########
    # Define the models
    mech_model = MechModel(1, params[muscle]['dedt_ce_max'], params[muscle]['kappa'], params['k_see'])
    energy_model = EnergeticsModel()

    #####
    # Perform the optimisation
    # Define the shortening values for the optimisation
    v_short_vals = -np.arange(0, 5, 0.5)  # s^{-1}

    # Compute the experimental energetics
    coeff = (params[muscle]['r_sl_b1996'], params[muscle]['r_am_b1996'])
    tot_heat_exp = np.polyval(coeff, -v_short_vals) # F0l0/s

    # Define the optimisation function
    def fun_opt(x):
        # Update values for parameters
        params[muscle]['r_cxb'] = x[0]
        params[muscle]['r_cat'] = x[1]
        params[muscle]['r_sl'] = x[2]

        # Loop over the shortening values
        mean_heat_tot = np.empty_like(v_short_vals, dtype=float)
        q_cat_mean = np.empty_like(v_short_vals, dtype=float)
        q_cxb_mean = np.empty_like(v_short_vals, dtype=float)
        q_sl_mean = np.empty_like(v_short_vals, dtype=float)

        for idx_v, v_short in enumerate(v_short_vals):
            
            # set up the mechanical conditions
            dedt_ce = np.zeros_like(t_vec) + v_short * (t_vec >= 1) * (t_vec < 1.1)
            # e_ce = cumtrapz(dedt_ce, t_vec, initial=1.05)
            e_ce = np.zeros_like(dedt_ce) # NOTE: We choose no strain since we are at steady state and shortening over plateau 

            # Get the force
            force = catn_vec * mech_model.F_va(dedt_ce) * mech_model.F_la(e_ce)
            
            # Compute the energy use 
            q_a, q_m, q_sl, w = energy_model.actEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force, mech_model)

            # Define the range over which to compute the energy use 
            t_range = (t_vec >= 1) * (t_vec < 1.1)

            # Compute the energy use from cxb compared to cat 
            q_cat_mean[idx_v] = np.mean(q_a[t_range])
            q_cxb_mean[idx_v] = np.mean(q_m[t_range])
            q_sl_mean[idx_v] = np.mean(q_sl[t_range])

            mean_heat_tot[idx_v] = np.mean(q_a[t_range] + q_m[t_range] + q_sl[t_range])

        # Return error with lagrange multiplier constrain implementation
        # return np.linalg.norm(tot_heat_exp - mean_heat_tot) / np.linalg.norm(tot_heat_exp)  + lambda_0 * (q_cxb_mean / q_cat_mean - 0.6 / 0.4) / (0.6 / 0.4)

        # Require that we match 60/40 split of cxb and cat heat 
        #   first two terms are the intercept 
        #   second term is the slope
        q_exp_iso = tot_heat_exp[0] 
        q_exp_cat = 0.4 * q_exp_iso
        q_exp_cxb = 0.6 * q_exp_iso 
        q_exp_short = tot_heat_exp - q_exp_iso
        # q_sl_mean = mean_heat_tot - q_cat_mean - q_cxb_mean
        # Smooth, normalized objective avoids the non-differenti~able abs() terms.
        scale = max(np.linalg.norm(tot_heat_exp), 1e-12)
        err_cat = (q_cat_mean - q_exp_cat) / scale
        err_cxb = (q_cxb_mean - q_exp_cxb) / scale
        err_short = (q_sl_mean - q_exp_short) / scale
        return np.mean(err_cat**2) + np.mean(err_cxb**2) + np.mean(err_short**2)
        # return np.abs(q_exp_cat - q_cat_mean[-1]) + np.abs(q_exp_cxb - q_cxb_mean[-1]) + np.linalg.norm(q_exp_short - q_sl_mean)

    # Perform basic optimisation 
    opt_res = minimize(fun_opt, x0=(params[muscle]['r_cxb'], params[muscle]['r_cat'], params[muscle]['r_sl']))

    # # Perform optimisation with random multi-start initialisation only.
    # bounds_ = ((0, 100), (0, 100), (0, 10), (1, 1))
    # n_starts = 20
    # rng = np.random.default_rng(42)

    # # Generate random initial guesses inside bounds.
    # x0_list = []
    # for _ in range(n_starts):
    #     x0_list.append(tuple(rng.uniform(low=b[0], high=b[1]) for b in bounds_))

    # best_res = None
    # for i_start, x0 in enumerate(x0_list, start=1):
    #     res = minimize(
    #         fun_opt,
    #         x0=x0,
    #         bounds=bounds_,
    #         options={'maxiter': 500, 'disp': False},
    #         method='Nelder-Mead'
    #     )

    #     if best_res is None or res.fun < best_res.fun:
    #         best_res = res

    #     print(f'start {i_start:02d}/{n_starts}: f = {res.fun:.6f}, x = {res.x}')

    # opt_res = best_res

    print(opt_res)
    params[muscle]['r_cxb'] = opt_res.x[0]
    params[muscle]['r_cat'] = opt_res.x[1]
    params[muscle]['r_sl'] = opt_res.x[2]

    print('Optimised parameters:')
    print(f'r_cxb = {params[muscle]["r_cxb"]}, r_cat = {params[muscle]["r_cat"]}, r_sl = {params[muscle]["r_sl"]}')

    #######
    # Plot a comparison of the optimised energetic rates

    mean_heat_tot = np.empty_like(v_short_vals, dtype=float)
    # fig_force, ax_force = plt.subplots(layout='constrained')
    # fig_force.suptitle(f'{muscle_type} - Force vs Time')
    # fig_strain, axs_strain = plt.subplots(2, 1, layout='constrained', figsize=(7, 6), sharex=True)
    # fig_strain.suptitle(f'{muscle_type} - Strain and Strain Rate')
    # fig_energy, axs_energy = plt.subplots(len(v_short_vals), 1, layout='constrained', figsize=(5, 20))
    # fig_energy.suptitle(f'{muscle_type} - Energy Rates')
    for idx_v, v_short in enumerate(v_short_vals):
        dedt_ce = np.zeros_like(t_vec) + v_short * (t_vec >= 1) * (t_vec < 1.1)
        # e_ce = cumtrapz(dedt_ce, t_vec, initial=1.05)
        e_ce = np.zeros_like(dedt_ce) # NOTE: We choose no strain since we are at steady state and shortening over plateau 

        # axs_strain[0].plot(t_vec, e_ce, label='$v_{short}$ = ' + str(v_short))
        # axs_strain[1].plot(t_vec, dedt_ce, label='$v_{short}$ = ' + str(v_short))

        force = catn_vec * mech_model.F_va(dedt_ce) * mech_model.F_la(e_ce)
        # ax_force.plot(t_vec, force, label='$v_{short}$ = ' + str(v_short))

        q_a, q_m, q_sl, w = energy_model.actEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force, mech_model)

        # ax_energy = axs_energy[idx_v]
        # ax_energy.plot(t_vec, q_a, label='$q_a$')
        # ax_energy.plot(t_vec, q_m, label='ddddddddddddddddd$q_m$')
        # ax_energy.plot(t_vec, q_sl, label='$q_{sl}$')
        # ax_energy.plot(t_vec, w, label='$w$')
        # ax_energy.legend()
        # ax_energy.set_xlabel('Time ($s$)')
        # ax_energy.set_ylabel('Energy rate ($F_0\,l_0\, s^{-1}$)')

        t_range = (t_vec >= 1) * (t_vec < 1.1)
        mean_heat_tot[idx_v] = np.mean(q_a[t_range] + q_m[t_range] + q_sl[t_range])

    # axs_strain[0].set_ylabel('$e_{ce}$')
    # axs_strain[0].grid()
    # axs_strain[1].set_xlabel('Time ($s$)')
    # axs_strain[1].set_ylabel('$\dot e_{ce}$ ($s^{-1}$)')
    # axs_strain[1].grid()
    # axs_strain[0].legend(loc='best')

    ax_hr.plot(v_short_vals, mean_heat_tot, label='Model: ' + muscle)
    ax_hr.plot(v_short_vals, tot_heat_exp, label='Experiment: ' + muscle)
    ax_hr.legend()
    ax_hr.grid()
    ax_hr.set_xticks(v_short_vals)
    ax_hr.set_xlabel('Shortening rate ($s^{-1}$)')
    ax_hr.set_ylabel('Mean energetic rate ($W (F_0 l_0)^{-1}$)')

fig_hr.savefig('Figures/1_cxbcatmech_fit.jpg')
fig_hr.savefig('Figures/1_cxbcatmech_fit.svg')

plt.show()


