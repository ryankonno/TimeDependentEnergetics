'''
Code to compare to the experimental protocol from Lewis and Barclay 2014. 

Data for the soleus and EDL are used and compared to the experimental data.

Ryan Konno
r.konno@uq.edu.au
The University of Queensland
'''

# Import 
import numpy as np 
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
import lib.plot_style

import sys 
sys.path.append('./')

# Import library 
from lib.recursive_merge import recursive_merge
from lib.model_metrics import r2_score, nrmse_calc

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
            't_stim_end': 0.125 # s, End time of stimulation, Barclay and Weber 2004

        }, 
        'EDL': {
            # Muscle-specific experimental protocol parameters
            'velo_short': 2.8, # l0/s, Barclay and Weber 2004
            'freq': 200, # Hz, Frequency of stimulation, Adjusted for tetenanus
            't_stim_end': 0.063 # s, End time of stimulation, Barclay and Weber 2004

        },
}

# Combine the dictionaries
params = recursive_merge(params_protocol, params_muscle)

# Compute additional parameters
for muscle in ('SOL', 'EDL'):
    # Maximum isometric force
    params[muscle]['F_0'] = (
        params[muscle]['mass'] / params['rho0'] /
        params[muscle]['l_0'] *
        params[muscle]['max_iso_stress']
    )
    print(f'{muscle}: Maximum isometric stress: {params[muscle]["F_0"]}')


def f_stim_length(t, params): 
    # Function to compute the length changes in the muscle 
    # returns both simulation times and lengths 

    cycle_length = params['cycle_length'] # s, Set at 0.3s for now (as in figure 1)
    t_cycle = t % cycle_length # Get the time with respect to the cycle
    N_cycles = params['N_cycles'] # Number of cycles to simulate

    # Time parameters edited to have a fixed stimulation time 
    if params['muscle'] == 'EDL':  
        t_stim_start = 0 
        t_stim_end = params[params['muscle']]['t_stim_end'] # From paper 
        t_short_start = 0.03 # Choose to get reasonable power & work... (sampe proportion as soleus values)
        # t_short_start = 0.005
        t_length_start = 0.15
        t_length_end =  cycle_length # Assume return to initial lenght by the end of the cycle
    elif params['muscle'] == 'SOL': 
        t_stim_start = 0 
        t_stim_end = params[params['muscle']]['t_stim_end'] # From paper
        t_short_start = 0.06
        t_length_start = 0.5
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

    # Get the stimulation times 
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


# Plot to verify conditions 
dt = 0.0001
t_vec = np.linspace(params['t_start'], params['t_end'], int((params['t_end'] - params['t_start']) / dt)) 

# Keep cycle frequency fixed at 1 Hz and sweep stimulation frequencies.
cycle_freq_hz = 1.0
stim_freq_list = (40, 80, 120, 160) # Hz, Stimulation frequencies

component_names = ('q_a', 'q_m', 'q_sl', 'w', 'q_r')
component_colors = ('#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e')

# Storage for cross-muscle comparison
peak_qr_by_muscle = {}
tau_by_muscle = {}
power_output_by_muscle = {}
initial_energy_rate_output_by_muscle = {}
total_energy_rate_output_by_muscle = {}
total_energy_output_by_muscle = {}
total_energy_output_mJg_by_muscle = {}

# Define plotting properties 
component_names = ('q_{cat}', 'q_{cxb}', 'q_{sl}', 'w', 'q_r')
hatch_styles = ('///', '\\\\', 'xx', '..', 'oo')
palette_cont_by_muscle = {
    'SOL': lib.plot_style.palette_cont_slow,
    'EDL': lib.plot_style.palette_cont_fast,
}

# Loop over muscles
for muscle_name in ('SOL', 'EDL'):
    print(f'\n========== {muscle_name} ==========')
    params['muscle'] = muscle_name
    
    # Initialise plots 
    fig_energy, ax_energy = plt.subplots(figsize=(4, 3))
    fig_energy.subplots_adjust(left=0.15)
    fig_force_cycle, ax_force_cycle = plt.subplots(figsize=(4, 3))
    fig_force_cycle.subplots_adjust(left=0.15)

    component_energy_abs = []
    peak_qr_vs_freq = []
    tau_vs_freq = []
    efficiency_rows = []
    total_energy_out_f0l0_vs_freq = []
    total_energy_out_mJg_vs_freq = []

    # Define the colour scheme 
    palette = palette_cont_by_muscle[muscle_name]

    for idx, stim_freq in enumerate(stim_freq_list): 
        cycle_length = 1.0 / cycle_freq_hz
        print(f'Stimulation frequency: {stim_freq} Hz')

        
        params['cycle_length'] = cycle_length
        params[muscle_name]['freq'] = stim_freq
        params['N_cycles'] = 10 # Fixed number of cycles (matches experimental conditions)
        stim_vec, stim_times_vec,  dl_vec = f_stim_length(t_vec, params)

        # Ca dynamics
        from Models.ActivationModel import ActivationModel
        act_model = ActivationModel(params[params['muscle']], t_vec)
        idx_stims = np.nonzero(stim_times_vec)[0]
        stim_vec, ca_vec, catn_vec = act_model.runExcAct(idx_stims, w_0 = 0.004)

        # Mechanics 
        from Models.MechanicsModel import MechModel 
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

        # Get the time for one cycle
        cycle_mask = (t_vec >= 0) * (t_vec < params['cycle_length'])
        t_cycle = t_vec[cycle_mask]

        # Compute the force directly  
        force_direct =  mech_model.computeForce(catn_vec, e_ce + 1, dedt_ce)
        force_cycle = force_direct[cycle_mask]

        # Plot the force
        ax_force_cycle.plot(t_cycle, force_cycle, color=palette[idx], label=f'{stim_freq} Hz')

        # Compute the initial energetics 
        from Models.InitialEnergeticsModel import EnergeticsModel
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
        from Models.BioenergeticsModel import Bioenergetics
        bioenergetic_model = Bioenergetics(params) 
        t_span = (t_vec[0], t_vec[-1]) 
        c_atp_0 = params[muscle]['c_atp_0']
        # Solve the model
        sol = bioenergetic_model.solveBioenergetics(t_span, c_atp_0, t_vec, E_initial_converted)
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
            q_a_cum = cumulative_trapezoid(
                q_a[cycle_mask], t_cycle, initial=0
            ) * energy_unit_scaler
            q_m_cum = cumulative_trapezoid(
                q_m[cycle_mask], t_cycle, initial=0
            ) * energy_unit_scaler
            q_sl_cum = cumulative_trapezoid(
                q_sl[cycle_mask], t_cycle, initial=0
            ) * energy_unit_scaler
            w_cum = cumulative_trapezoid(
                w[cycle_mask], t_cycle, initial=0
            ) * energy_unit_scaler
            q_r_cum = cumulative_trapezoid(
                q_r[cycle_mask], t_cycle, initial=0
            ) * energy_unit_scaler
            
        # Plot total energy
        ax_energy.plot(
            t_vec,
            cumulative_trapezoid(E_tot, t_vec, initial=0) *
            energy_unit_scaler,
            label='$e_{init}$', color=palette[idx], alpha=0.25
        )
        ax_energy.plot(
            t_vec,
            cumulative_trapezoid(q_r, t_vec, initial=0) *
            energy_unit_scaler,
            label='$q_r$', color=palette[idx], ls=lib.plot_style.ls_styles[2],
            alpha=0.5
        )
        ax_energy.plot(
            t_vec,
            cumulative_trapezoid(E_tot + q_r, t_vec, initial=0) *
            energy_unit_scaler,
            label='$q_r + e_{init}$', color=palette[idx]
        ) 
        ax_energy.set_xlabel('Time (s)')
        ax_energy.set_ylabel('Energy  ($mJ g^{-1}$)')
        ax_energy.grid(True, alpha=0.3)

        # # Compute energetics using previous model
        # energy_rate_data, energy_data_ = energy_model.dHdt_Konno2025(
        #     catn_vec, t_vec, e_ce, dedt_ce,
        #     force_direct, params[muscle]['r1'],
        #     params[muscle]['r2'], params
        # )
        # E_tot_konno2025 = energy_rate_data['dEdt']
        # ax_energy.plot(
        #     t_vec,
        #     cumulative_trapezoid(E_tot_konno2025, t_vec, initial=0) /
        #     params[muscle]['mass'] * 1e3,
        #     label='KLD2025 Model', color='k'
        # ) 

        # Store absolute end-of-trial energies for component contribution bar charts.
        e_q_a_end = cumulative_trapezoid(q_a, t_vec, initial = 0)[-1] * energy_unit_scaler
        e_q_m_end = cumulative_trapezoid(q_m, t_vec, initial = 0)[-1] * energy_unit_scaler
        e_q_sl_end = cumulative_trapezoid(q_sl, t_vec, initial = 0)[-1] * energy_unit_scaler
        e_w_end = cumulative_trapezoid(w, t_vec, initial = 0)[-1] * energy_unit_scaler
        e_q_r_end = cumulative_trapezoid(q_r, t_vec, initial = 0)[-1] * energy_unit_scaler
        component_energy_abs.append((e_q_a_end, e_q_m_end, e_q_sl_end, e_w_end, e_q_r_end))

        #######################
        # Compute the time constants from the data 
        total_energy_rate = (E_tot + q_r) * energy_unit_scaler
        mask = t_vec >= params['cycle_length'] * params['N_cycles'] + 3 # 3s buffer
        t_decay = t_vec[mask]
        y_decay = total_energy_rate[mask]
        t_rel = t_decay - t_decay[0]

        def exp_decay(t, y_inf, A, tau):
            return y_inf + A * np.exp(-t / tau)

        # Match 2_runsim_B1995 initialisation and bounds for exponential fitting.
        tail_n = min(500, len(y_decay))
        y_inf_guess = float(np.mean(y_decay[-tail_n:]))
        A_guess = float(y_decay[0] - y_inf_guess)
        tau_guess = 20.0

        p0 = (y_inf_guess, A_guess, tau_guess)
        bounds = ([-np.inf, -np.inf, 1e-9], [np.inf, np.inf, np.inf])
        popt, _ = curve_fit(exp_decay, t_rel, y_decay, p0=p0, bounds=bounds, maxfev=20000)
        y_inf_fit, A_fit, tau_fit = popt

        # Compute the peak recovery rate 
        peak_qr_vs_freq.append(np.max(q_r[mask] * energy_unit_scaler))
        tau_vs_freq.append(tau_fit)

        # Compute efficiencies from integrated energies over the full simulation window.
        E_tot_end = cumulative_trapezoid(E_tot, t_vec, initial = 0)[-1]
        E_rec_end = cumulative_trapezoid(q_r, t_vec, initial = 0)[-1]
        W_end = cumulative_trapezoid(w, t_vec, initial = 0)[-1]
        E_tot_end_mJg = E_tot_end * energy_unit_scaler
        E_rec_end_mJg = E_rec_end * energy_unit_scaler
        E_total_out_mJg = E_tot_end_mJg + E_rec_end_mJg
        E_total_out_F0l0 = (
            E_total_out_mJg *
            params[muscle]['mass'] /
            params[muscle]['F_0'] /
            params[muscle]['l_0'] * 1e-3
        )
        total_energy_out_f0l0_vs_freq.append(E_total_out_F0l0)
        total_energy_out_mJg_vs_freq.append(E_total_out_mJg)
        n_contractions = params['N_cycles']
        if n_contractions > 0 and cycle_freq_hz > 0:
            # Compute average per-contraction output, then multiply by contraction frequency.
            E_init_per_contraction_mJg = E_tot_end_mJg / n_contractions
            E_total_per_contraction_mJg = E_total_out_mJg / n_contractions
            W_per_contraction_mJg = (W_end * energy_unit_scaler) / n_contractions

            E_init_rate_out_mWg = E_init_per_contraction_mJg * cycle_freq_hz
            E_total_rate_out_mWg = E_total_per_contraction_mJg * cycle_freq_hz
            power_out_mWg = W_per_contraction_mJg * cycle_freq_hz
        else:
            E_init_rate_out_mWg = np.nan
            E_total_rate_out_mWg = np.nan
            power_out_mWg = np.nan

        eta_init = W_end / E_tot_end if np.abs(E_tot_end) > 0 else np.nan
        eta_total = (
            W_end / (E_tot_end + E_rec_end)
            if np.abs(E_tot_end + E_rec_end) > 0
            else np.nan
        )
        efficiency_ratio = (
            (eta_init / eta_total - 1)
            if (not np.isnan(eta_total) and eta_total > 0)
            else np.nan
        )
        efficiency_rows.append((
            stim_freq, eta_init, eta_total, efficiency_ratio,
            E_tot_end_mJg, E_rec_end_mJg,
            E_init_rate_out_mWg, E_total_rate_out_mWg,
            power_out_mWg
        ))

        print(f'Fitted time constant (tau) = {tau_fit:.3f} s')

    ax_force_cycle.set_xlabel('Time within cycle (s)')
    ax_force_cycle.set_ylabel('Force (N)')
    ax_force_cycle.set_xlim((0,params[muscle]['t_stim_end'] + 0.5))
    ax_force_cycle.grid(True, alpha=0.3)
    ax_force_cycle.legend(loc = 'upper right')

    # Plot component contributions vs frequency (absolute energies at trial end)
    component_energy_abs = np.array(component_energy_abs)
    x = np.arange(len(stim_freq_list))
    n_comp = len(component_names)
    bar_width = 0.15

    # Plot relative component contributions (normalised by total end-of-trial energy)
    row_totals = np.sum(component_energy_abs, axis = 1)
    component_energy_rel = component_energy_abs / row_totals[:, None]

    fig_comp_rel, ax_comp_rel = plt.subplots(figsize=(4, 3))
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
            color=component_colors[comp_idx],
            label=component_names[comp_idx]
        )
        q_i_bottom += component_energy_rel[:, comp_idx]

    # Separate q_r bar for each condition
    ax_comp_rel.bar(
        x_qr,
        component_energy_rel[:, 4],
        width=bar_width_rel,
        color=component_colors[4],
        label=component_names[4]
    )
    ax_comp_rel.set_title(f'{muscle_name} - Relative Contributions')
    ax_comp_rel.set_xticks(x)
    ax_comp_rel.set_xticklabels([f'{stim_freq} Hz' for stim_freq in stim_freq_list])
    ax_comp_rel.set_xlabel('Stimulation frequency (Hz)')
    ax_comp_rel.set_ylabel('Relative contribution at trial end')
    ax_comp_rel.grid(True, alpha=0.3)
    # ax_comp_rel.legend()

    peak_qr_by_muscle[muscle_name] = np.array(peak_qr_vs_freq)
    tau_by_muscle[muscle_name] = np.array(tau_vs_freq)
    total_energy_output_by_muscle[muscle_name] = np.array(total_energy_out_f0l0_vs_freq)
    total_energy_output_mJg_by_muscle[muscle_name] = np.array(total_energy_out_mJg_vs_freq)

    # Print efficiency table in terminal for the current muscle.
    print(f'\n{muscle_name} efficiency table')
    print('freq_Hz | eta_init | eta_total | E_rec/E_tot |'
          ' E_init | E_rec | E_init_rate | E_total_rate'
          ' | P_out')
    for (freq, eta_init, eta_total, efficiency_ratio,
         E_tot_end_mJg, E_rec_end_mJg, E_init_rate_out_mWg,
         E_total_rate_out_mWg, power_out_mWg) in efficiency_rows:
        print(f'{freq:7.2f} | {eta_init:9.6f} |'
              f'{eta_total:10.6f} | {efficiency_ratio:11.6f} |'
              f'{E_tot_end_mJg:7.3f} | {E_rec_end_mJg:6.3f} |'
              f'{E_init_rate_out_mWg:11.3f} | '
              f'{E_total_rate_out_mWg:12.3f} | '
              f'{power_out_mWg:6.3f}')

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
    # fig_energy.savefig('Figures/L2014_energy_' + muscle + '.jpg')
    # fig_energy.savefig('Figures/L2014_energy_' + muscle + '.svg')
    # Save the per-muscle force cycle figure
    # fig_force_cycle.savefig('Figures/L2014_force_' + muscle + '.jpg')
    # fig_force_cycle.savefig('Figures/L2014_force_' + muscle + '.svg')


# Plot initial and total energy-rate output against power output for both muscles.
fig_energy_power_compare, ax_energy_power_compare = plt.subplots(figsize=(4, 3))
fig_energy_power_compare.subplots_adjust(left=0.15)
ax_energy_power_compare.plot(
    power_output_by_muscle['SOL'],
    initial_energy_rate_output_by_muscle['SOL'],
    ':o', label='SOL initial rate', color=lib.plot_style.palette_cont_slow[0]
)
ax_energy_power_compare.plot(
    power_output_by_muscle['SOL'],
    total_energy_rate_output_by_muscle['SOL'],
    '-o', label='SOL total rate', color=lib.plot_style.palette_cont_slow[0]
)
ax_energy_power_compare.plot(
    power_output_by_muscle['EDL'],
    initial_energy_rate_output_by_muscle['EDL'],
    ':o', label='EDL initial rate', color=lib.plot_style.palette_cont_fast[0]
)
ax_energy_power_compare.plot(
    power_output_by_muscle['EDL'],
    total_energy_rate_output_by_muscle['EDL'],
    '-o', label='EDL total rate', color=lib.plot_style.palette_cont_fast[0]
)
ax_energy_power_compare.set_xlabel('Power output ($mW g^{-1}$)')
ax_energy_power_compare.set_ylabel('Total energy rate output ($mW g^{-1}$)')
ax_energy_power_compare.grid(True, alpha = 0.3)
ax_energy_power_compare.legend(loc = 'upper right')
fig_energy_power_compare.savefig('Figures/L2014_energy_vs_power_comp.jpg')
fig_energy_power_compare.savefig('Figures/L2014_energy_vs_power_comp.svg')

# Plot total energy output in mJ g^-1 for both muscles.
fig_total_energy_mJg_compare, ax_total_energy_mJg_compare = plt.subplots(figsize=(4, 3))
fig_total_energy_mJg_compare.subplots_adjust(left=0.15)
ax_total_energy_mJg_compare.plot(
    stim_freq_list, total_energy_output_mJg_by_muscle['SOL'],
    '-o', label='SOL', color=lib.plot_style.palette_cont_slow[0]
)
ax_total_energy_mJg_compare.plot(
    stim_freq_list, total_energy_output_mJg_by_muscle['EDL'],
    '-o', label='EDL', color=lib.plot_style.palette_cont_fast[0]
)

exp_energy_sol = np.genfromtxt('Data/L2014_data_stimfreq_energy_SOL.csv', delimiter=',', names=True)
exp_energy_edl = np.genfromtxt('Data/L2014_data_stimfreq_energy_EDL.csv', delimiter=',', names=True)
ax_total_energy_mJg_compare.plot(
    exp_energy_sol['freq'], exp_energy_sol['E'],
    ':o', lw=1.5, ms=4, label='EXP SOL', color=lib.plot_style.palette_cont_slow[0]
)
ax_total_energy_mJg_compare.plot(
    exp_energy_edl['freq'], exp_energy_edl['E'],
    ':o', lw=1.5, ms=4, label='EXP EDL', color=lib.plot_style.palette_cont_fast[0]
)

ax_total_energy_mJg_compare.set_xlabel('Stimulation frequency (Hz)')
ax_total_energy_mJg_compare.set_ylabel('Total energy ($mJ g^{-1}$)')
ax_total_energy_mJg_compare.grid(True, alpha = 0.3)
ax_total_energy_mJg_compare.legend(loc = 'upper right')
fig_total_energy_mJg_compare.savefig('Figures/L2014_total_energy_mJg_comp.jpg')
fig_total_energy_mJg_compare.savefig('Figures/L2014_total_energy_mJg_comp.svg')

# Plot total energy output normalised by each series maximum.
def normalise_to_max(values):
    values = np.asarray(values, dtype=float)
    max_val = np.max(values)
    if max_val <= 0:
        return values
    return values / max_val

fig_total_energy_mJg_compare_norm, ax_total_energy_mJg_compare_norm = plt.subplots(figsize=(4, 3))
fig_total_energy_mJg_compare_norm.subplots_adjust(left=0.15)

sol_model_norm = normalise_to_max(total_energy_output_mJg_by_muscle['SOL'])
edl_model_norm = normalise_to_max(total_energy_output_mJg_by_muscle['EDL'])
sol_exp_norm = normalise_to_max(exp_energy_sol['E'])
edl_exp_norm = normalise_to_max(exp_energy_edl['E'])

ax_total_energy_mJg_compare_norm.plot(
    stim_freq_list, sol_model_norm,
    '-o', label='SOL', color=lib.plot_style.palette_cont_slow[0]
)
ax_total_energy_mJg_compare_norm.plot(
    stim_freq_list, edl_model_norm,
    '-o', label='EDL', color=lib.plot_style.palette_cont_fast[0]
)
ax_total_energy_mJg_compare_norm.plot(
    exp_energy_sol['freq'], sol_exp_norm,
    ':o', lw=1.5, ms=4, label='EXP SOL', color=lib.plot_style.palette_cont_slow[0]
)
ax_total_energy_mJg_compare_norm.plot(
    exp_energy_edl['freq'], edl_exp_norm,
    ':o', lw=1.5, ms=4, label='EXP EDL', color=lib.plot_style.palette_cont_fast[0]
)

ax_total_energy_mJg_compare_norm.set_xlabel('Stimulation frequency (Hz)')
ax_total_energy_mJg_compare_norm.set_ylabel('Normalised total energy (-)')
ax_total_energy_mJg_compare_norm.grid(True, alpha = 0.3)
ax_total_energy_mJg_compare_norm.legend(loc = 'upper right')
fig_total_energy_mJg_compare_norm.savefig('Figures/L2014_total_energy_mJg_comp_norm.jpg')
fig_total_energy_mJg_compare_norm.savefig('Figures/L2014_total_energy_mJg_comp_norm.svg')

# Compute the r^2 values of the fit 
# Interpolate 
mod_vals_interp_sol = np.interp(exp_energy_sol['freq'], stim_freq_list, total_energy_output_mJg_by_muscle['SOL'])
mod_vals_interp_edl = np.interp(exp_energy_edl['freq'], stim_freq_list, total_energy_output_mJg_by_muscle['EDL'])
# Compute normalised r2 values 
r2_sol = r2_score(mod_vals_interp_sol/max(mod_vals_interp_sol), exp_energy_sol['E']/max( exp_energy_sol['E']))
r2_edl = r2_score(mod_vals_interp_edl/max(mod_vals_interp_edl), exp_energy_edl['E']/max(exp_energy_edl['E']))
print(f'SOL: r2 = {r2_sol}')
print(f'EDL: r2 = {r2_edl}')
mse_model_sol = nrmse_calc(mod_vals_interp_sol/max(mod_vals_interp_sol), exp_energy_sol['E']/max( exp_energy_sol['E']))
mse_model_edl = nrmse_calc(mod_vals_interp_edl/max(mod_vals_interp_edl), exp_energy_edl['E']/max(exp_energy_edl['E']))
print(f'SOL: nrmse = {mse_model_sol}')
print(f'EDL: nrmse = {mse_model_edl}')

plt.show()
