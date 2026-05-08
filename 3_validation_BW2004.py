'''
This code simulates protocol from the Barclay and Weber 2004 experiments. 

Both SOL and EDL data are used and results are compared to the experimental data. 

Ryan Konno
r.konno@uq.edu.au
The University of Queensland
'''

# Import 
import numpy as np 
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit

# Import matplotlib and define plot properties
import matplotlib.pyplot as plt 
import lib.plot_style

# Define colour schemes
# palette_cont = ("#E69F00","#56B4E9","#009E73","#F0E442","#0072B2", "#CC79A7")
palette_cont_slow = lib.plot_style.palette_cont_slow
palette_cont_fast = lib.plot_style.palette_cont_fast
# palette = ("#32cd9c", "#f67410", "#2b21b8", "#C21599", "#83d921", "#1ab6e9")
ls_styles = lib.plot_style.ls_styles

import sys 
sys.path.append('./')

from lib.recursive_merge import recursive_merge

# Import models
from Models.BioenergeticsModel import Bioenergetics
from Models.MechanicsModel import MechModel
from Models.ActivationModel import ActivationModel
from Models.InitialEnergeticsModel import EnergeticsModel

# Import parameters 
from parameters_muscle import params as params_muscle

'''
Define protocol specific parmeters 
'''
params_protocol = {
    # Time parameters for setting up the protocol 
    't_start': 0, # s
    't_end': 70, # s
    'N_cycles': 10, # unitless, Number of cycles to simulate (rest period after N_cycles contractions)

    # General muscle parameters
    'rho0':  1e6, # g/m^3, Density of muscle

    'muscle': 'SOL', # Specify muscle parameters to be used in simulation
        # Mouse data 
        'SOL': {
            # Muscle-specific experimental protocol parameters 
            'velo_short': 1.3, # l0/s, Barclay and Weber 2004
            'freq': 150, # Hz, Frequency of stimulation, adjusted for tetanus

        }, 
        'EDL': {
            # Muscle-specific experimental protocol parameters
            'velo_short': 2.8, # l0/s, Barclay and Weber 2004
            'freq': 175, # Hz, Frequency of stimulation, Adjusted for tetenanus

        },
}

# Combine the dictionaries
params = recursive_merge(params_protocol, params_muscle)

# Compute parameter values not in the dictionaries
for muscle in ('SOL', 'EDL'):
    # Maximum isometric force
    params[muscle]['F_0'] = (
        params[muscle]['mass'] / params['rho0'] /
        params[muscle]['l_0'] *
        params[muscle]['max_iso_stress']
    )
    print(f'{muscle}: Maximum isometric force: {params[muscle]["F_0"]}')

'''
Setup the simulation 
'''
def f_stim_length(t, params): 
    # Function to compute the length changes in the muscle 
    # returns both simulation times and lengths 

    cycle_length = params['cycle_length'] # s, Set at 0.3s for now (as in figure 1)
    t_cycle = t % cycle_length # Get the time with respect to the cycle
    N_cycles = params['N_cycles'] # Number of cycles to simulate

    # Time parameters edited to have a fixed stimulation time 
    if params['muscle'] == 'EDL':  
        t_stim_start = 0 
        t_stim_end = 0.063 # From paper 
        t_short_start = 0.03 # Choose to get reasonable power & work... (sampe proportion as soleus values)
        # t_short_start = 0.005
        t_length_start = 0.15
        t_length_end =  cycle_length # Assume return to initial lenght by the end of the cycle
    elif params['muscle'] == 'SOL': 
        t_stim_start = 0 
        t_stim_end = 0.125 # From paper
        t_short_start = 0.06
        t_length_start = 0.245
        t_length_end =  cycle_length # Assume return to initial lenght by the end of the cycle

    # Get the optimal length of the muscle
    l_0 = params[params['muscle']]['l_0']

    # Fix the shortening rate across conditions (s^-1)
    v_short = -params[params['muscle']]['velo_short'] * l_0

    # Compute the lengthening velocity (s^-1)
    v_length = (
        (- v_short * (t_length_start - t_short_start)) /
        (t_length_end - t_length_start)
    )  

    # Compute the change in length (m)
    dl = (
        (
            (t_cycle > t_short_start) * (t_cycle < t_length_start) *
            (v_short * (t_cycle - t_short_start)) +
            (t_cycle >= t_length_start) * (t_cycle < t_length_end) *
            (v_short * (t_length_start - t_short_start) +
             v_length * (t_cycle - t_length_start))
        ) *
        (t < cycle_length * N_cycles)
    )

    # Toggle whether in stimulation or not (does not define frequency of stim here)
    stim = ((t_cycle >= t_stim_start) * (t_cycle <= t_stim_end) ) * (t < cycle_length * N_cycles)

    # Compute the stimulation times 
    # stim_times: vector (same shape as t) with 1 where a stimulus (spike) occurs, 0 otherwise
    stim_times = np.zeros_like(t, dtype=int)

    # Get stimulation frequency from params for the active muscle
    freq = params[params['muscle']]['freq']

    # Get teh stimulation times 
    if freq > 0:
        period = 1.0 / freq

        # Get times when there is a stimulus 
        t_stim_period = t[stim] 

        # Get the firing times
        t_fire_vec = []
        if t_stim_period.size > 0:
            t_fire_prev = t_stim_period[0]
            t_fire_vec.append(t_fire_prev)
            # iterate remaining times and add a spike when period elapsed
            for t_val in t_stim_period[1:]:
                if t_val >= t_fire_prev + period - 1e-12:
                    t_fire_vec.append(t_val)
                    t_fire_prev = t_val
        t_fire_vec = np.array(t_fire_vec)

        # Map each spike time to the nearest index in t and mark stim_times
        stim_times = np.zeros_like(t, dtype=int)
        if t_fire_vec.size > 0:
            t_arr = np.asarray(t)
            for st in t_fire_vec:
                idx = int(np.argmin(np.abs(t_arr - st)))
                stim_times[idx] = 1
            
    return stim, stim_times, dl


# Define the time vector 
dt = 0.0001
t_vec = np.linspace(params['t_start'], params['t_end'], int((params['t_end'] - params['t_start']) / dt)) 

# Define frequencies for the cycles, Hz
freq_list = (0.5, 1, 2, 3, 4) 

# Define plotting properties 
component_names = ('q_{cat}', 'q_{cxb}', 'q_{sl}', 'w', 'q_r')
hatch_styles = ('///', '\\\\', 'xx', '..', 'oo')
palette_cont_by_muscle = {
    'SOL': palette_cont_slow,
    'EDL': palette_cont_fast,
}

# Storage for cross-muscle comparison
peak_qr_by_muscle = {}
tau_by_muscle = {}
power_output_by_muscle = {}
initial_energy_rate_output_by_muscle = {}
total_energy_rate_output_by_muscle = {}
total_energy_output_by_muscle = {}
trace_plot_data_by_muscle = {}

# Loop over muscles
for muscle_name in ('SOL', 'EDL'):
    print(f'\n========== {muscle_name} ==========')
    params['muscle'] = muscle_name
    palette_cont = palette_cont_by_muscle[muscle_name]
    
    # Initialise plots 
    fig_energy, ax_energy = plt.subplots()
    fig_energy.subplots_adjust(left=0.15)
    fig_tau, ax_tau = plt.subplots()
    fig_tau.subplots_adjust(left=0.15)
    fig_strain_cycle, axs_strain_cycle = plt.subplots(2, 1)
    fig_strain_cycle.subplots_adjust(left=0.15)
    fig_force_cycle, ax_force_cycle = plt.subplots()
    fig_force_cycle.subplots_adjust(left=0.15)

    
    # Figure for time-varying energy components during one cycle
    fig_energy_components, ax_energy_comp = plt.subplots(
        figsize=(4, 3)
    )
    fig_energy_components.subplots_adjust(left=0.15)

    component_energy_abs = []
    peak_qr_vs_freq = []
    tau_vs_freq = []
    efficiency_rows = []
    total_energy_out_f0l0_vs_freq = []

    for idx, freq in enumerate(freq_list):
        print(f'Frequency: {freq} Hz')

        stim_length = 1/freq # s, stimulation time         
        params['cycle_length'] = stim_length
        params['N_cycles'] = 10 # Fixed number of cycles (matches experimental conditions)
        stim_vec, stim_times_vec,  dl_vec = f_stim_length(t_vec, params)

        # Ca dynamics
        act_model = ActivationModel(params[params['muscle']], t_vec)
        idx_stims = np.nonzero(stim_times_vec)[0]
        stim_vec, ca_vec, catn_vec = act_model.runExcAct(idx_stims, w_0 = 0.004)

        # Mechanics 
        muscle = params['muscle']
        mech_model = MechModel(
            params[muscle]['l_0'],
            params[muscle]['dedt_ce_max'],
            params[muscle]['kappa'],
            params['k_see']
        )

        # Compute strain and strain rates in muscle
        e_ce = dl_vec / params[muscle]['l_0'] + 0.1
        dedt_ce = np.diff(e_ce, prepend = 0) / np.diff(t_vec, prepend = 1)

        # Plot one cycle for the current frequency condition.
        cycle_mask = (t_vec >= 0) * (t_vec < params['cycle_length'])
        t_cycle = t_vec[cycle_mask]
        e_ce_cycle = e_ce[cycle_mask]
        dedt_ce_cycle = dedt_ce[cycle_mask]
        axs_strain_cycle[0].plot(t_cycle, e_ce_cycle, color=palette_cont[idx], label=f'{freq} Hz')
        axs_strain_cycle[1].plot(t_cycle, dedt_ce_cycle, color=palette_cont[idx], label=f'{freq} Hz')

        # Compute the force directly  
        force_direct =  mech_model.computeForce(catn_vec, e_ce + 1, dedt_ce)
        force_cycle = force_direct[cycle_mask]
        ax_force_cycle.plot(t_cycle, force_cycle, color=palette_cont[idx], label=f'{freq} Hz')

        # Store trace data for first frequency (for cross-muscle comparison)
        if idx == 0:
            trace_plot_data_by_muscle[muscle_name] = {
                't_cycle': t_cycle,
                'e_ce_cycle': e_ce_cycle,
                'stim_cycle': stim_vec[cycle_mask],
                'force_cycle': force_cycle
            }

        # Compute the initial energetics 
        energy_model = EnergeticsModel()
        q_a, q_m, q_sl, w = energy_model.solveInitialEnergetics(
            t_vec, ca_vec, catn_vec, params[muscle],
            e_ce + 1, dedt_ce, force_direct, mech_model
        )
        E_tot = q_a + q_m + q_sl + w  # F0l0/s, Total energy 

        # Convert units to input for bioenergetics
        E_initial_converted = (
            E_tot * params[muscle]['F_0'] *
            params[muscle]['l_0'] / params[muscle]['mass']
        )

        # Run bioenergetics
        bioenergetic_model = Bioenergetics(params) 
        t_span = (t_vec[0], t_vec[-1]) 
        c_atp_0 = params[muscle]['c_atp_0']
        # Solve the model
        sol = bioenergetic_model.solveBioenergetics(t_span, c_atp_0, t_vec, E_initial_converted)

        # Compute and plot gamma(t) from the bioenergetics model.
        c_adp_vec = np.maximum(bioenergetic_model.c_a_tot - sol.y[0,], 0.0)

        # Compute the energetic rates 
        scale = (
            params[muscle]['mass'] / params[muscle]['F_0'] /
            params[muscle]['l_0']
        )
        q_r = (
            bioenergetic_model.computeRecoveryEnergetics(
                sol.t, sol.y[0,]
            ) * scale
        )

        # Compute unit scaler
        energy_unit_scaler = (
            params[muscle]['F_0'] * params[muscle]['l_0'] /
            params[muscle]['mass'] * 1e3
        )

        # Plot time-varying energy components for first frequency
        if idx == 0:
            # Extract one cycle
            cycle_mask = (t_vec >= 0) * (t_vec < params['cycle_length'])
            t_cycle = t_vec[cycle_mask]
            
            # Compute cumulative energy for each component
            q_a_cum = cumtrapz(
                q_a[cycle_mask], t_cycle, initial=0
            ) * energy_unit_scaler
            q_m_cum = cumtrapz(
                q_m[cycle_mask], t_cycle, initial=0
            ) * energy_unit_scaler
            q_sl_cum = cumtrapz(
                q_sl[cycle_mask], t_cycle, initial=0
            ) * energy_unit_scaler
            w_cum = cumtrapz(
                w[cycle_mask], t_cycle, initial=0
            ) * energy_unit_scaler
            q_r_cum = cumtrapz(
                q_r[cycle_mask], t_cycle, initial=0
            ) * energy_unit_scaler
            
            # Plot components on single axes
            ax_energy_comp.plot(
                t_cycle, q_a_cum, label='$q_a$ (activation)',
                color=palette_cont[0], ls = ls_styles[0], linewidth=2
            )
            ax_energy_comp.plot(
                t_cycle, q_m_cum, label='$q_m$ (maintenance)',
                color=palette_cont[0], ls = ls_styles[1], linewidth=2
            )
            ax_energy_comp.plot(
                t_cycle, q_sl_cum, label='$q_{sl}$ (shortening)',
                color=palette_cont[0], ls = ls_styles[2], linewidth=2
            )
            ax_energy_comp.plot(
                t_cycle, w_cum, label='$w$ (mechanical work)',
                color=palette_cont[0], ls = ls_styles[3], linewidth=2
            )
            ax_energy_comp.plot(
                t_cycle, q_r_cum, label='$q_r$ (recovery)',
                color=palette_cont[0], ls = ls_styles[4], linewidth=2
            )
            
            # Add labels and formatting
            ax_energy_comp.set_xlabel('Time (s)')
            ax_energy_comp.set_ylabel('Cumulative Energy (mJ/g)')
            ax_energy_comp.legend(loc='upper left')
            ax_energy_comp.grid(True, alpha=0.3)

        # Plot total energy
        ax_energy.plot(
            t_vec,
            cumtrapz(E_tot, t_vec, initial=0) *
            energy_unit_scaler,
            label='$e_{init}$', color=palette_cont[idx], alpha=0.25
        )
        ax_energy.plot(
            t_vec,
            cumtrapz(q_r, t_vec, initial=0) *
            energy_unit_scaler,
            label='$q_r$', color=palette_cont[idx], ls=':',
            alpha=0.5
        )
        ax_energy.plot(
            t_vec,
            cumtrapz(E_tot + q_r, t_vec, initial=0) *
            energy_unit_scaler,
            label='$q_r + e_{init}$', color=palette_cont[idx]
        ) 
        ax_energy.set_xlabel('Time (s)')
        ax_energy.set_ylabel('Energy ($mJ g^{-1}$)')

        # Compute energetics using previous model
        energy_rate_data, energy_data_ = energy_model.dHdt_Konno2025(
            catn_vec, t_vec, e_ce, dedt_ce,
            force_direct, params[muscle]['r1'],
            params[muscle]['r2'], params
        )
        E_tot_konno2025 = energy_rate_data['dEdt']
        ax_energy.plot(
            t_vec,
            cumtrapz(E_tot_konno2025, t_vec, initial=0) /
            params[muscle]['mass'] * 1e3,
            label='KLD2025 Model', color='k'
        ) 

        # Store absolute end-of-trial energies for component contribution bar charts
        e_q_a_end = cumtrapz(q_a, t_vec, initial = 0)[-1] * energy_unit_scaler
        e_q_m_end = cumtrapz(q_m, t_vec, initial = 0)[-1] * energy_unit_scaler
        e_q_sl_end = cumtrapz(q_sl, t_vec, initial = 0)[-1] * energy_unit_scaler
        e_w_end = cumtrapz(w, t_vec, initial = 0)[-1] * energy_unit_scaler
        e_q_r_end = cumtrapz(q_r, t_vec, initial = 0)[-1] * energy_unit_scaler
        component_energy_abs.append((e_q_a_end, e_q_m_end, e_q_sl_end, e_w_end, e_q_r_end))

        #######################
        # Compute the time constants from the data 
        total_energy_rate = (E_tot + q_r) * energy_unit_scaler
        mask = t_vec >= params['cycle_length'] * params['N_cycles'] + 3 # 3s buffer
        t_decay = t_vec[mask]
        y_decay = total_energy_rate[mask]
        t_rel = t_decay - t_decay[0]

        # Define function for the exponential decay
        def exp_decay(t, y_inf, A, tau):
            return y_inf + A * np.exp(-t / tau)

        tail_n = min(500, len(y_decay))
        y_inf_guess = float(np.mean(y_decay[-tail_n:]))
        A_guess = float(y_decay[0] - y_inf_guess)
        tau_guess = 20.0

        p0 = (y_inf_guess, A_guess, tau_guess)
        bounds = ([-np.inf, -np.inf, 1e-9], [np.inf, np.inf, np.inf])
        popt, _ = curve_fit(exp_decay, t_rel, y_decay, p0=p0, bounds=bounds, maxfev=20000)
        y_inf_fit, A_fit, tau_fit = popt

        ax_tau.plot(t_rel, y_decay, color = palette_cont[idx], alpha = 0.35, label = f'{freq} Hz decay')
        ax_tau.plot(
            t_rel, exp_decay(t_rel, *popt), '--',
            color=palette_cont[idx],
            label=f'{freq} Hz fit ($\\tau$ = '
                  f'{tau_fit:.2f} s)'
        )

        # Compute the peak recovery rate 
        peak_qr_vs_freq.append(np.max(q_r[mask] * energy_unit_scaler))
        tau_vs_freq.append(tau_fit)

        # Compute efficiencies from integrated energies over the full simulation window.
        E_tot_end = cumtrapz(E_tot, t_vec, initial = 0)[-1]
        E_rec_end = cumtrapz(q_r, t_vec, initial = 0)[-1]
        W_end = cumtrapz(w, t_vec, initial = 0)[-1]
        E_tot_end_mJg = E_tot_end * energy_unit_scaler
        E_rec_end_mJg = E_rec_end * energy_unit_scaler
        W_end_mJg = W_end * energy_unit_scaler
        E_total_out_mJg = E_tot_end_mJg + E_rec_end_mJg
        E_total_out_F0l0 = (
            E_total_out_mJg *
            params[muscle]['mass'] /
            params[muscle]['F_0'] /
            params[muscle]['l_0'] * 1e-3
        )
        total_energy_out_f0l0_vs_freq.append(E_total_out_F0l0)
        n_contractions = params['N_cycles']
        if n_contractions > 0 and freq > 0:
            # Compute average per-contraction output, then multiply by contraction frequency.
            E_init_per_contraction_mJg = E_tot_end_mJg / n_contractions
            E_total_per_contraction_mJg = E_total_out_mJg / n_contractions
            W_per_contraction_mJg = (W_end * energy_unit_scaler) / n_contractions

            E_init_rate_out_mWg = E_init_per_contraction_mJg * freq
            E_total_rate_out_mWg = E_total_per_contraction_mJg * freq
            power_out_mWg = W_per_contraction_mJg * freq
        else:
            E_init_rate_out_mWg = np.nan
            E_total_rate_out_mWg = np.nan
            power_out_mWg = np.nan

        eta_init = W_end / E_tot_end
        eta_total = W_end / (E_tot_end + E_rec_end)
        efficiency_ratio = eta_init / eta_total - 1

        # Combine into data structure
        efficiency_rows.append((
            freq, eta_init, eta_total, efficiency_ratio,
            E_tot_end_mJg, E_rec_end_mJg,
            E_init_rate_out_mWg, E_total_rate_out_mWg,
            power_out_mWg, W_end_mJg
        ))

        print(f'Fitted time constant (tau) = {tau_fit:.3f} s')

    ax_tau.set_xlabel('Time since end of stimulation (s)')
    ax_tau.set_ylabel('Total energy rate ($mW g^{-1}$)')
    ax_tau.legend(loc = 'upper right')

    axs_strain_cycle[0].set_xlabel('Time within cycle (s)')
    axs_strain_cycle[0].set_ylabel('$e_{ce}$')
    axs_strain_cycle[0].grid(True, alpha=0.3)
    axs_strain_cycle[1].set_xlabel('Time within cycle (s)')
    axs_strain_cycle[1].set_ylabel('$\dot e_{ce}$ ($s^{-1}$)')
    axs_strain_cycle[1].grid(True, alpha=0.3)

    ax_force_cycle.set_xlabel('Time within cycle (s)')
    ax_force_cycle.set_ylabel('Force (N)')
    ax_force_cycle.grid(True, alpha=0.3)
    ax_force_cycle.legend(loc = 'upper right')

    # Plot relative component contributions (normalised by total end-of-trial energy)
    component_energy_abs = np.array(component_energy_abs)
    x = np.arange(len(freq_list))
    n_comp = len(component_names)
    bar_width_rel = 0.35
    row_totals = np.sum(component_energy_abs, axis=1)
    component_energy_rel = component_energy_abs / row_totals[:, None]

    fig_comp_rel, ax_comp_rel = plt.subplots()
    fig_comp_rel.subplots_adjust(left=0.15)
    bar_width_rel = 0.35
    x_qi = x - bar_width_rel / 2
    x_qr = x + bar_width_rel / 2

    # Stacked q_i bar for each condition: q_a + q_m + q_sl + w
    q_i_bottom = np.zeros_like(x, dtype=float)
    for comp_idx in range(4):
        ax_comp_rel.bar(
            x_qi,
            component_energy_rel[:, comp_idx],
            width = bar_width_rel,
            bottom=q_i_bottom,
            facecolor='white',
            edgecolor='black',
            hatch=hatch_styles[comp_idx],
            label=component_names[comp_idx]
        )
        q_i_bottom += component_energy_rel[:, comp_idx]

    # Separate q_r bar for each condition
    ax_comp_rel.bar(
        x_qr,
        component_energy_rel[:, 4],
        width=bar_width_rel,
        facecolor='white',
        edgecolor='black',
        hatch=hatch_styles[4],
        label=component_names[4]
    )
    # ax_comp_rel.set_title(f'{muscle_name} - Relative Contributions')
    ax_comp_rel.set_xticks(x)
    ax_comp_rel.set_xticklabels([f'{freq} Hz' for freq in freq_list])
    ax_comp_rel.set_xlabel('Cycle frequency')
    ax_comp_rel.set_ylabel('Relative contribution at trial end')
    # ax_comp_rel.legend()

    peak_qr_by_muscle[muscle_name] = np.array(peak_qr_vs_freq)
    tau_by_muscle[muscle_name] = np.array(tau_vs_freq)
    total_energy_output_by_muscle[muscle_name] = np.array(total_energy_out_f0l0_vs_freq)

    # Print efficiency table in terminal for the current muscle.
    print(f'\n{muscle_name} efficiency table')
    print('freq_Hz | eta_init | eta_total | E_rec/E_tot |'
          ' E_init | E_rec | E_init_rate | E_total_rate'
          ' | P_out | W_total')
    for (freq, eta_init, eta_total, efficiency_ratio,
         E_tot_end_mJg, E_rec_end_mJg, E_init_rate_out_mWg,
         E_total_rate_out_mWg, power_out_mWg, W_end_mJg) in efficiency_rows:
        print(f'{freq:7.2f} | {eta_init:9.6f} |'
              f'{eta_total:10.6f} | {efficiency_ratio:11.6f} |'
              f'{E_tot_end_mJg:7.3f} | {E_rec_end_mJg:6.3f} |'
              f'{E_init_rate_out_mWg:11.3f} | '
              f'{E_total_rate_out_mWg:12.3f} | '
              f'{power_out_mWg:6.3f} | {W_end_mJg:8.3f}')

    efficiency_array = np.array(efficiency_rows, dtype=float)
    mean_vals = np.nanmean(efficiency_array[:, 1:], axis=0)
    sem_vals = (
        np.nanstd(efficiency_array[:, 1:], axis=0, ddof=1) /
        np.sqrt(np.sum(
            ~np.isnan(efficiency_array[:, 1:]), axis=0
        ))
    )
    print(' mean±SEM | '
          f'{mean_vals[0]:.6f} ± {sem_vals[0]:.6f} | '
          f'{mean_vals[1]:.6f} ± {sem_vals[1]:.6f} | '
            f'{mean_vals[2]:.6f} ± {sem_vals[2]:.6f} | '
            f'{mean_vals[3]:.6f} ± {sem_vals[3]:.6f} | '
            f'{mean_vals[4]:.6f} ± {sem_vals[4]:.6f} | '
            f'{mean_vals[5]:.6f} ± {sem_vals[5]:.6f} | '
            f'{mean_vals[6]:.6f} ± {sem_vals[6]:.6f}')

    # Save for cross-muscle energy-vs-power comparison.
    efficiency_array = np.array(efficiency_rows, dtype=float)
    initial_energy_rate_output_by_muscle[muscle_name] = efficiency_array[:, 6]
    total_energy_rate_output_by_muscle[muscle_name] = efficiency_array[:, 7]
    power_output_by_muscle[muscle_name] = efficiency_array[:, 8]
        
    # Save the energetics figure 
    fig_energy.savefig('Figures/B2004_energy_' + muscle + '.jpg')
    fig_energy.savefig('Figures/B2004_energy_' + muscle + '.svg')
    fig_energy_components.savefig(
        'Figures/B2004_energy_components_' + muscle + '.jpg'
    )
    fig_energy_components.savefig(
        'Figures/B2004_energy_components_' + muscle + '.svg'
    )


'''
Comparison plots with data from both muscles 
'''
# Create strain and force trace plots with both muscles overlaid
if 'SOL' in trace_plot_data_by_muscle and 'EDL' in trace_plot_data_by_muscle:
    sol_data = trace_plot_data_by_muscle['SOL']
    edl_data = trace_plot_data_by_muscle['EDL']
    sol_color = palette_cont_slow[0]
    edl_color = palette_cont_fast[0]
    
    # Create strain figure
    fig_trace_strain, ax_trace_strain = plt.subplots(figsize=(4, 3))
    fig_trace_strain.subplots_adjust(left=0.15)
    ax_trace_strain.plot(sol_data['t_cycle'], sol_data['e_ce_cycle'], label='SOL', color=sol_color, linewidth=2)
    ax_trace_strain.plot(edl_data['t_cycle'], edl_data['e_ce_cycle'], label='EDL', color=edl_color, linewidth=2)
    
    # Add stimulation region (use SOL data for timing)
    stim_mask_sol = sol_data['stim_cycle'] > 0
    # Add stimulation regions for both muscles
    stim_mask_edl = edl_data['stim_cycle'] > 0
    if np.any(stim_mask_sol):
        t_stim_sol = sol_data['t_cycle'][stim_mask_sol]
        ax_trace_strain.axvspan(t_stim_sol[0], t_stim_sol[-1], color=sol_color, alpha=0.15, label='Stim (SOL)')
    if np.any(stim_mask_edl):
        t_stim_edl = edl_data['t_cycle'][stim_mask_edl]
        ax_trace_strain.axvspan(t_stim_edl[0], t_stim_edl[-1], color=edl_color, alpha=0.15, label='Stim (EDL)')
    
    ax_trace_strain.set_xlabel('Time within cycle (s)')
    ax_trace_strain.set_ylabel('Strain ($e_{ce}$)')
    ax_trace_strain.grid(True, alpha=0.3)
    ax_trace_strain.legend(loc='upper right')
    fig_trace_strain.savefig('Figures/B2004_trace_strain_comp.jpg')
    fig_trace_strain.savefig('Figures/B2004_trace_strain_comp.svg')
    
    # Create force figure
    fig_trace_force, ax_trace_force = plt.subplots(figsize=(4, 3))
    fig_trace_force.subplots_adjust(left=0.15)
    ax_trace_force.plot(sol_data['t_cycle'], sol_data['force_cycle'], label='SOL', color=sol_color, linewidth=2)
    ax_trace_force.plot(edl_data['t_cycle'], edl_data['force_cycle'], label='EDL', color=edl_color, linewidth=2)
    
    # Add stimulation region
    stim_mask_edl = edl_data['stim_cycle'] > 0
    if np.any(stim_mask_sol):
        t_stim_sol = sol_data['t_cycle'][stim_mask_sol]
        ax_trace_force.axvspan(t_stim_sol[0], t_stim_sol[-1], color=sol_color, alpha=0.15, label='Stim (SOL)')
    if np.any(stim_mask_edl):
        t_stim_edl = edl_data['t_cycle'][stim_mask_edl]
        ax_trace_force.axvspan(t_stim_edl[0], t_stim_edl[-1], color=edl_color, alpha=0.15, label='Stim (EDL)')
    
    ax_trace_force.set_xlabel('Time within cycle (s)')
    ax_trace_force.set_ylabel('Force (N)')
    ax_trace_force.grid(True, alpha=0.3)
    ax_trace_force.legend(loc='upper right')
    fig_trace_force.savefig('Figures/B2004_trace_force_comp.jpg')
    fig_trace_force.savefig('Figures/B2004_trace_force_comp.svg')

# Plot peak recovery rate versus frequency for both muscles
fig_peak_qr_compare, ax_peak_qr_compare = plt.subplots()
fig_peak_qr_compare.subplots_adjust(left=0.15)
ax_peak_qr_compare.plot(freq_list, peak_qr_by_muscle['SOL'], '-o', label='SOL', color=palette_cont_slow[0])
ax_peak_qr_compare.plot(freq_list, peak_qr_by_muscle['EDL'], '-o', label='EDL', color=palette_cont_fast[0])
ax_peak_qr_compare.set_xlabel('Cycle frequency (Hz)')
ax_peak_qr_compare.set_ylabel('Peak recovery rate ($mW g^{-1}$)')

exp_rrecmax_sol = np.genfromtxt('Data/BW2004_data_rrecmax_SOL.csv', delimiter=',', names=True)
exp_rrecmax_edl = np.genfromtxt('Data/BW2004_data_rrecmax_EDL.csv', delimiter=',', names=True)
ax_peak_qr_compare.plot(
    exp_rrecmax_sol['freq'][0:len(freq_list)],
    exp_rrecmax_sol['rrecmax'][0:len(freq_list)],
    'k-o', lw=1.5, ms=4, label='EXP'
)
ax_peak_qr_compare.plot(
    exp_rrecmax_edl['freq'][0:len(freq_list)],
    exp_rrecmax_edl['rrecmax'][0:len(freq_list)],
    'k--s', lw=1.5, ms=4, label='_nolegend_'
)

ax_peak_qr_compare.grid(True, alpha = 0.3)
ax_peak_qr_compare.legend(loc = 'upper right')
fig_peak_qr_compare.savefig('Figures/B2004_SepVars_rrecmax_comp.jpg')
fig_peak_qr_compare.savefig('Figures/B2004_SepVars_rrecmax_comp.svg')

# Plot initial and total energy-rate output against power output for both muscles.
fig_energy_power_compare, ax_energy_power_compare = plt.subplots()
fig_energy_power_compare.subplots_adjust(left=0.15)
ax_energy_power_compare.plot(
    power_output_by_muscle['SOL'],
    initial_energy_rate_output_by_muscle['SOL'],
    '--o', label='SOL initial rate', color=palette_cont_slow[0]
)
ax_energy_power_compare.plot(
    power_output_by_muscle['SOL'],
    total_energy_rate_output_by_muscle['SOL'],
    '-o', label='SOL total rate', color=palette_cont_slow[0]
)
ax_energy_power_compare.plot(
    power_output_by_muscle['EDL'],
    initial_energy_rate_output_by_muscle['EDL'],
    '--o', label='EDL initial rate', color=palette_cont_fast[0]
)
ax_energy_power_compare.plot(
    power_output_by_muscle['EDL'],
    total_energy_rate_output_by_muscle['EDL'],
    '-o', label='EDL total rate', color=palette_cont_fast[0]
)
ax_energy_power_compare.set_xlabel('Power output ($mW g^{-1}$)')
ax_energy_power_compare.set_ylabel('Total energy rate output ($mW g^{-1}$)')
ax_energy_power_compare.grid(True, alpha = 0.3)
ax_energy_power_compare.legend(loc = 'upper right')
fig_energy_power_compare.savefig('Figures/B2004_energy_vs_power_comp.jpg')
fig_energy_power_compare.savefig('Figures/B2004_energy_vs_power_comp.svg')

# Plot fitted time constants versus frequency for both muscles
fig_tau_freq_compare, ax_tau_freq_compare = plt.subplots()
fig_tau_freq_compare.subplots_adjust(left=0.15)
ax_tau_freq_compare.plot(freq_list, tau_by_muscle['SOL'], '-o', label='SOL', color=palette_cont_slow[0])
ax_tau_freq_compare.plot(freq_list, tau_by_muscle['EDL'], '-o', label='EDL', color=palette_cont_fast[0])
ax_tau_freq_compare.set_xlabel('Cycle frequency (Hz)')
ax_tau_freq_compare.set_ylabel('Time constant $\\tau$ (s)')

exp_tau_sol = np.genfromtxt('Data/BW2004_data_tau_SOL.csv', delimiter=',', names=True)
exp_tau_edl = np.genfromtxt('Data/BW2004_data_tau_EDL.csv', delimiter=',', names=True)
ax_tau_freq_compare.plot(
    exp_tau_sol['freq'][0:len(freq_list)],
    exp_tau_sol['tau'][0:len(freq_list)],
    'k-o', lw=1.5, ms=4, label='EXP'
)
ax_tau_freq_compare.plot(
    exp_tau_edl['freq'][0:len(freq_list)],
    exp_tau_edl['tau'][0:len(freq_list)],
    'k--s', lw=1.5, ms=4, label='_nolegend_'
)

ax_tau_freq_compare.grid(True, alpha = 0.3)
ax_tau_freq_compare.legend(loc = 'upper right')

fig_tau_freq_compare.savefig('Figures/B2004_SepVars_tau_comp.jpg')
fig_tau_freq_compare.savefig('Figures/B2004_SepVars_tau_comp.svg')

# Compute mean experimental and model time constants
print('\n========== TIME CONSTANT ANALYSIS ==========')
for muscle_name in ('SOL', 'EDL'):
    # Load experimental tau data
    if muscle_name == 'SOL':
        exp_tau_data = np.genfromtxt('Data/BW2004_data_tau_SOL.csv', delimiter=',', names=True)
    else:
        exp_tau_data = np.genfromtxt('Data/BW2004_data_tau_EDL.csv', delimiter=',', names=True)
    
    # Extract experimental tau values for the frequencies we simulated
    exp_tau_values = exp_tau_data['tau'][0:len(freq_list)]
    model_tau_values = tau_by_muscle[muscle_name]
    
    # Compute means and standard errors
    mean_exp_tau = np.mean(exp_tau_values)
    sem_exp_tau = np.std(exp_tau_values, ddof=1) / np.sqrt(len(exp_tau_values))
    
    mean_model_tau = np.mean(model_tau_values)
    sem_model_tau = np.std(model_tau_values, ddof=1) / np.sqrt(len(model_tau_values))
    
    # Compute percent difference
    percent_diff = ((mean_model_tau - mean_exp_tau) / mean_exp_tau) * 100
    
    print(f'\n{muscle_name}:')
    print(f'  Mean experimental tau: {mean_exp_tau:.4f} ± {sem_exp_tau:.4f} s')
    print(f'  Mean model tau:        {mean_model_tau:.4f} ± {sem_model_tau:.4f} s')
    print(f'  Percent difference:    {percent_diff:.2f}%')

plt.show()
