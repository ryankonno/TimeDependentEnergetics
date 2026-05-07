'''
Code to determine the initial model parameters.

Ryan Konno
r.konno@uq.edu.au
The University of Queensland 
'''

# Import 
import numpy as np 
from scipy.integrate import cumtrapz
from scipy.optimize import minimize
import matplotlib.pyplot as plt 
import lib.plot_style
import pandas as pd
import sys 
sys.path.append('./')
from lib.recursive_merge import recursive_merge

# Import muscle parameters 
from parameters_muscle import params as params_muscle

# Define parameter values 
params_protocol = {
    # Time parameters for setting up the protocol 
    't_start': 0, # s
    't_end': 2, # s

    # General muscle parameters
    'rho0':  1e6, # g/m^3, Density of muscle

    'muscle': 'SOL', # Specify muscle parameters to be used in simulation
        # Mouse data 
        'SOL': {
            'freq': 150, # Hz, Frequency of stimulation , 
            # Heat rate parameters Barclay 1996
            # 'r_am_exp': 0.6177, # W/F_0/l_0, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_sl_exp': 0.234, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)
            # heat parameters from Barclay 2010
            'r_am_exp': 0.38, # W/F_0/l_0, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_sl_exp': 0.11, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

        }, 
        'EDL': {
            'freq': 150, # Hz, Frequency of stimulation 
            #  # Heat parameters from Barclay 1996
            # 'r_am_exp': 2.792, # F0l0/s, Maximum heat rate of isometric contraction (fast-type fibre)
            # 'r_sl_exp': 0.697, # F0l0/s, Maximum shortening heat rate (fast-type fibre)
            # heat parameters from Barclay 2010
            'r_am_exp': 1.13, # W/F_0/l_0, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_sl_exp': 0.07, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

        },
}

# Combine the dictionaries
params = recursive_merge(params_protocol, params_muscle)

# Data files for each muscle type (SOL = soleus/slow, EDL = fast)
data_files = {
    'SOL': 'Data/Barclay1996_sol_heatrates_data.txt',
    'EDL': 'Data/Barclay1996_edl_heatrates_data.txt',
}

# Load the models
from Models.ActivationModel import ActivationModel
from Models.MechanicsModel import MechModel 
from Models.InitialEnergeticsModel import EnergeticsModel

# Initialise plots for the model 
fig_hr, ax_hr = plt.subplots(layout='constrained')


########
# Loop over both muscles: SOL (soleus/slow) and EDL (fast)
for muscle_type in ['SOL','EDL']:
    params['muscle'] = muscle_type
    muscle = muscle_type

    # Load the experimental data for this muscle
    data_exp = pd.read_csv(data_files[muscle_type])

    print(f'\n=== Processing {muscle_type} ({"slow/soleus" if muscle_type == "SOL" else "fast/EDL"}) ===')

    t_vec = np.linspace(params['t_start'], params['t_end'], int(50000 * params['t_end']))

    ########
    # Define the models
    mech_model = MechModel(1, params[muscle]['dedt_ce_max'], params[muscle]['kappa'], params['k_see'])
    energy_model = EnergeticsModel()

    #####
    # Perform the optimisation
    # Define the shortening values for the optimisation
    v_short_vals = -np.arange(0, 5, 1)  # 1/s

    # Compute the experimental energetics
    coeff = (params[muscle]['r_sl_exp'], params[muscle]['r_am_exp'])
    tot_heat_exp = np.polyval(coeff, -v_short_vals) # F0l0/s

    # Define dictionary to keep track of optimiser progress
    progress_state = {
        'iter': 0,
        'last': None,
    }

    # Define the optimisation function
    def fun_opt(x):
        # Update values for parameters
        params[muscle]['r_cxb'] = x[0]
        params[muscle]['r_cat'] = x[1]
        params[muscle]['cxb_scale'] = x[2]
        params[muscle]['r_sl'] = x[3]

        # Loop over the shortening values
        mean_heat_tot = np.empty_like(v_short_vals, dtype=float)
        q_cat_mean = np.empty_like(v_short_vals, dtype=float)
        q_cxb_mean = np.empty_like(v_short_vals, dtype=float)
        q_sl_mean = np.empty_like(v_short_vals, dtype=float)

        ########
        # Simulate the Ca dynamics and mechanics (constant for all shortening rates)
        # compute the ca dynamics for all contractions (same across conditions)
        
        act_model = ActivationModel(params[muscle], t_vec)
        # Compute the stimulus times
        period = 1.0 / params[muscle]['freq']
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

        for idx_v, v_short in enumerate(v_short_vals):
            
            # set up the mechanical conditions
            dedt_ce = np.zeros_like(t_vec) + v_short * (t_vec >= 1) * (t_vec < 1.1)
            # e_ce = cumtrapz(dedt_ce, t_vec, initial=1.05)
            e_ce = np.zeros_like(dedt_ce) # NOTE: We choose no strain since we are at steady state and shortening over plateau 

            # Get the force
            force = catn_vec * mech_model.F_va(dedt_ce) * mech_model.F_la(e_ce + 1)
            
            # Compute the energy use 
            q_a, q_m, q_sl, w = energy_model.solveInitialEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force, mech_model)

            # Define the range over which to compute the energy use 
            t_range = (t_vec >= 1) * (t_vec < 1.1)

            # Compute the energy use from cxb compared to cat 
            q_cat_mean[idx_v] = np.mean(q_a[t_range])
            q_cxb_mean[idx_v] = np.mean(q_m[t_range])
            q_sl_mean[idx_v] = np.mean(q_sl[t_range])

            mean_heat_tot[idx_v] = np.mean(q_a[t_range] + q_m[t_range] + q_sl[t_range])

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
        
        #________________________________________
        # Calculation enforcing the dependence on frequency 
        # We only look at the isometric condition

        # Define the frequencies
        freq_opt_list = np.linspace(1,160,5) # Similar to Lewis and Barclay 2014

        # Initialise the error in the ratio
        error_ratio = 0

        for idx_v, freq_stim in enumerate(freq_opt_list):
            # Create a new activation model
            act_model = ActivationModel(params[params['muscle']], t_vec)
            # Compute the stimulus times
            period = 1.0 / freq_stim
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
            
            # set up the mechanical conditions
            dedt_ce = np.zeros_like(t_vec)
            e_ce = np.zeros_like(dedt_ce) # We choose no strain since we are at steady state and shortening over plateau 

            # Get the force
            force = catn_vec * mech_model.F_va(dedt_ce) * mech_model.F_la(e_ce + 1)
            
            # Compute the energy use 
            q_a, q_m, q_sl, w = energy_model.solveInitialEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force, mech_model)
            assert sum(w) == 0 # enforce no work condition
            q_tot = q_a + q_m
            
            # Integrate 
            q_a_int = np.trapz(q_a, t_vec)
            q_tot_int = np.trapz(q_tot, t_vec)
        
            # Compute the error in the ratio of  cross-bridge heat to initial energy
            error_ratio += (q_a_int / q_tot_int - 0.4)**2

        err_cat_mse = np.mean(err_cat**2)
        err_cxb_mse = np.mean(err_cxb**2)
        err_short_mse = np.mean(err_short**2)
        err_ratio_penalty = np.mean(error_ratio)
        objective = err_cat_mse + err_cxb_mse + err_short_mse + err_ratio_penalty

        # Save the progress in the optimiser
        progress_state['last'] = {
            'x': np.array(x, dtype=float).copy(),
            'err_cat_mse': float(err_cat_mse),
            'err_cxb_mse': float(err_cxb_mse),
            'err_short_mse': float(err_short_mse),
            'err_ratio_penalty': float(err_ratio_penalty),
            'error_ratio_raw': float(error_ratio),
            'objective': float(objective),
        }

        return objective

    # Define the callback function
    def callback_fun(xk):
        progress_state['iter'] += 1
        it = progress_state['iter']

        # Report the most recent objective decomposition computed by fun_opt.
        last = progress_state.get('last')
        if last is None:
            print(
                f"[{muscle_type}] Iter {it:04d} | "
                f"r_cxb={xk[0]:.6g}, r_cat={xk[1]:.6g}, "
                f"cxb_scale={xk[2]:.6g}, r_sl={xk[3]:.6g}"
            )
            return

        print(
            f"[{muscle_type}] Iter {it:04d} | "
            f"r_cxb={xk[0]:.6g}, r_cat={xk[1]:.6g}, "
            f"cxb_scale={xk[2]:.6g}, r_sl={xk[3]:.6g} | "
            f"err_cat={last['err_cat_mse']:.3e}, "
            f"err_cxb={last['err_cxb_mse']:.3e}, "
            f"err_short={last['err_short_mse']:.3e}, "
            f"err_ratio_penalty={last['err_ratio_penalty']:.3e}, "
            f"error_ratio_raw={last['error_ratio_raw']:.3e}, "
            f"obj={last['objective']:.3e}"
        )

    # Perform a bounded optimisation 
    cxb_scale_0 = np.clip(params[muscle]['cxb_scale'], 0.55, 0.95)
    r_sl_0 = np.clip(params[muscle]['r_sl'], 0.01, 0.5)
    x0 = (params[muscle]['r_cxb'], params[muscle]['r_cat'], cxb_scale_0, r_sl_0)
    bounds_ = ((0.001,4), (0.0001, 4), (0.01,1), (0.0001, 2))
    # Define the initial simplex
    initial_simplex = np.array([
        [x0[0], x0[1], x0[2], x0[3]],
        [min(x0[0] + 0.1, bounds_[0][1]), x0[1], x0[2], x0[3]],
        [x0[0], min(x0[1] + 0.1, bounds_[1][1]), x0[2], x0[3]],
        [x0[0], x0[1], min(x0[2] + 0.05, bounds_[2][1]), x0[3]],
        [x0[0], x0[1], x0[2], min(x0[3] + 0.05, bounds_[3][1])],
    ])

    # Perform the optimisation
    opt_res = minimize(
        fun_opt,
        x0,
        callback=callback_fun,
        bounds=bounds_,
        options={'maxiter': 500, 'disp': True, 'initial_simplex': initial_simplex, 'xatol': 1e-4, 'fatol': 1e-4},
        method='Nelder-Mead'
    )

    # Print out the optimal solution 
    x = opt_res.x 
    print(f'r_cxb = {x[0]}, r_cat = {x[1]}, cxb_scale = {x[2]}, r_sl = {x[3]}')
    params[muscle]['r_cxb'], params[muscle]['r_cat'], params[muscle]['cxb_scale'], params[muscle]['r_sl'] = x[0], x[1], x[2], x[3]

    print(opt_res)
    params[muscle]['r_cxb'] = opt_res.x[0]
    params[muscle]['r_cat'] = opt_res.x[1]
    params[muscle]['cxb_scale'] = opt_res.x[2]
    params[muscle]['r_sl'] = opt_res.x[3]

    print('Optimised parameters:')
    print(f'r_cxb = {params[muscle]["r_cxb"]}, r_cat = {params[muscle]["r_cat"]}, cxb_scale = { params[muscle]["cxb_scale"]}, r_sl = { params[muscle]["r_sl"]}')

    #######
    # Plot a comparison of the optimised energetic rates and force traces

    # Create array to store mean heat values 
    mean_heat_tot = np.empty_like(v_short_vals, dtype=float)

    # Initialise figures 
    fig_force, ax_force = plt.subplots(layout='constrained')
    fig_force.suptitle(f'{muscle_type} - Force vs Time')
    fig_energy_components, axs_energy_comp = plt.subplots(
        5, 1, layout='constrained', figsize=(10, 12)
    )

    # Loop over the velocities and re compute the model energy predictions + plot
    velocity_colors = plt.cm.viridis(np.linspace(0, 1, len(v_short_vals)))
    for idx_v, v_short in enumerate(v_short_vals):

        # Get the mechanical state
        dedt_ce = np.zeros_like(t_vec) + v_short * (t_vec >= 1) * (t_vec < 1.1)
        e_ce = np.zeros_like(dedt_ce) # NOTE: We choose no strain since we are at steady state and shortening over plateau 
        
        # INitialise activation model 
        act_model = ActivationModel(params[params['muscle']], t_vec)

        # Compute the stimulus times
        period = 1.0 / params[params['muscle']]['freq']
        # Get the firing times
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
        
        # Compute the muscle force
        force = catn_vec * mech_model.F_va(dedt_ce) * mech_model.F_la(e_ce + 1)
        ax_force.plot(t_vec, force, label='$v_{short}$ = ' + str(v_short))

        # Solve the intitial energetics model 
        q_a, q_m, q_sl, w = energy_model.solveInitialEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force, mech_model)

        # Compute cumulative energy for each component
        q_a_cum = cumtrapz(q_a, t_vec, initial=0)
        q_m_cum = cumtrapz(q_m, t_vec, initial=0)
        q_sl_cum = cumtrapz(q_sl, t_vec, initial=0)
        w_cum = cumtrapz(w, t_vec, initial=0)
        total_cum = cumtrapz(q_a + q_m + q_sl + w, t_vec, initial=0)

        # Overlay each shortening value on the same 5-panel figure
        axs_energy_comp[0].plot(
            t_vec, q_a_cum,
            label=f'$v_{{short}}$ = {v_short} l$_0$/s',
            color=velocity_colors[idx_v], linewidth=2
        )
        axs_energy_comp[1].plot(
            t_vec, q_m_cum,
            label=f'$v_{{short}}$ = {v_short} l$_0$/s',
            color=velocity_colors[idx_v], linewidth=2
        )
        axs_energy_comp[2].plot(
            t_vec, q_sl_cum,
            label=f'$v_{{short}}$ = {v_short} l$_0$/s',
            color=velocity_colors[idx_v], linewidth=2
        )
        axs_energy_comp[3].plot(
            t_vec, w_cum,
            label=f'$v_{{short}}$ = {v_short} l$_0$/s',
            color=velocity_colors[idx_v], linewidth=2
        )
        axs_energy_comp[4].plot(
            t_vec, total_cum,
            label=f'$v_{{short}}$ = {v_short} l$_0$/s',
            color=velocity_colors[idx_v], linewidth=2
        )

        # Add labels and formatting once after all curves are drawn
        axs_energy_comp[0].set_ylabel('Activation\nEnergy (F$_0$l$_0$)')
        axs_energy_comp[1].set_ylabel('Maintenance\nEnergy (F$_0$l$_0$)')
        axs_energy_comp[2].set_ylabel('Shortening\nEnergy (F$_0$l$_0$)')
        axs_energy_comp[3].set_ylabel('Work\n(F$_0$l$_0$)')
        axs_energy_comp[4].set_ylabel('Total\nEnergy (F$_0$l$_0$)')
        for ax in axs_energy_comp:
            ax.grid(True, alpha=0.3)
        axs_energy_comp[-1].set_xlabel('Time (s)')
        fig_energy_components.suptitle(
            f'{muscle_type}: cumulative energy components',
            fontsize=12, fontweight='bold'
        )

        t_range = (t_vec >= 1) * (t_vec < 1.1)
        mean_heat_tot[idx_v] = np.mean(q_a[t_range] + q_m[t_range] + q_sl[t_range])
    
    # Plot the comparison to experimental results
    ax_hr.plot(v_short_vals, mean_heat_tot, label='Model: ' + muscle)
    ax_hr.plot(v_short_vals, tot_heat_exp, label='Experiment: ' + muscle)
    ax_hr.legend()
    ax_hr.grid()
    ax_hr.set_xticks(v_short_vals)
    ax_hr.set_xlabel('Shortening rate ($s^{-1}$)')
    ax_hr.set_ylabel('Mean energetic rate ($W (F_0 l_0)^{-1}$)')

    # Update force labels and save 
    ax_force.set_xlabel('Time ($s$)')
    ax_force.set_ylabel('Force (N)')
    ax_force.grid(True, alpha=0.3)
    ax_force.legend(loc='best')
    fig_force.savefig(f'Figures/1_cxbcatmech_force_{muscle}.jpg')
    fig_force.savefig(f'Figures/1_cxbcatmech_force_{muscle}.svg')
    
    # Save energy components figure
    fig_energy_components.savefig(
        f'Figures/1_cxbcatmech_energy_components_{muscle}.jpg'
    )
    fig_energy_components.savefig(
        f'Figures/1_cxbcatmech_energy_components_{muscle}.svg'
    )

    # Plot activation-to-total heat ratio across frequencies
    #   need to recompute the energetics for a range of frequencies
    # Initialise the plot
    fig_ratio, ax_ratio = plt.subplots(layout='constrained')
    fig_ratio.suptitle(f'{muscle_type}: activation/total heat ratio vs frequency')
    freq_plot_vec = np.linspace(40, 160, 25)
    ratio_plot_vec = np.zeros_like(freq_plot_vec, dtype=float)
    for idx_f, freq_stim in enumerate(freq_plot_vec):
        # Re initialise activation model
        act_model = ActivationModel(params[params['muscle']], t_vec)

        # Get the stimulation times for the frequency
        period = 1.0 / freq_stim
        t_fire_vec = []
        if t_vec.size > 0:
            t_fire_prev = t_vec[0]
            t_fire_vec.append(t_fire_prev)
            for t_val in t_vec[1:]:
                if t_val >= t_fire_prev + period - 1e-12:
                    t_fire_vec.append(t_val)
                    t_fire_prev = t_val
        t_fire_vec = np.array(t_fire_vec)

        stim_times = np.zeros_like(t_vec, dtype=int)
        if t_fire_vec.size > 0:
            t_arr = np.asarray(t_vec)
            for st in t_fire_vec:
                idx = int(np.argmin(np.abs(t_arr - st)))
                stim_times[idx] = 1

        # Solve the activation model
        idx_stims = np.nonzero(stim_times)[0]
        stim_vec, ca_vec, catn_vec = act_model.runExcAct(idx_stims)

        # Compute the mechanics
        dedt_ce = np.zeros_like(t_vec)
        e_ce = np.zeros_like(dedt_ce)
        force = catn_vec * mech_model.F_va(dedt_ce) * mech_model.F_la(e_ce + 1)

        # Compute the energetics
        q_a, q_m, q_sl, w = energy_model.solveInitialEnergetics(
            t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force, mech_model
        )

        q_a_int = np.trapz(q_a, t_vec)
        q_tot_int = np.trapz(q_a + q_m + q_sl + w, t_vec)

        # Save the ratio
        ratio_plot_vec[idx_f] = q_a_int / q_tot_int

    # Plot and save
    ax_ratio.plot(freq_plot_vec, ratio_plot_vec, linewidth=2, label=f'Model: {muscle}')
    ax_ratio.set_xlabel('Stimulation frequency (Hz)')
    ax_ratio.set_ylabel('Activation/Total heat ratio')
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.legend(loc='best')
    fig_ratio.savefig(f'Figures/1_cxbcatmech_activation_ratio_vs_frequency_{muscle}.jpg')
    fig_ratio.savefig(f'Figures/1_cxbcatmech_activation_ratio_vs_frequency_{muscle}.svg')

fig_hr.savefig('Figures/1_cxbcatmech_fit.jpg')
fig_hr.savefig('Figures/1_cxbcatmech_fit.svg')

plt.show()


