'''
B2004 validation with sensitivity analysis 

Run sensitivity analysis where to address the possible role of temperature on the parameter values 

Ryan Konno
'''

# Import 
import numpy as np 
from scipy.integrate import solve_ivp, cumtrapz
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import minimize, minimize_scalar,curve_fit
import matplotlib.pyplot as plt 
font = {'size'   : 14}
plt.rc('font', **font)
import matplotlib.cm as cmap
palette = ("#32cd9c", "#f67410", "#2b21b8", "#C21599", "#83d921", "#1ab6e9")
import itertools
import sys 
import copy
sys.path.append('./')
# Parameters
params = {
    # Time parameters for setting up the protocol 
    't_start': 0, # s
    't_end': 70, # s
    'cycle_length': 0.3, # s, Defines the length of the cycle (used to set frequency of the contractions)
    'N_cycles': 10, # unitless, Number of cycles to simulate (rest period after N_cycles contractions)

    # General muscle parameters
    'rho0':  1e6,    # g/m^3, Density of muscle
    'max_iso_stress': 2.5e5, # N/m^2, Maximum isometric stress of the muscle

    'muscle': 'SOL', # Specify muscle parameters to be used in simulation

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

            #__________
            # Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, BUGFIXED!
            # 'V_max_oxphos': 0.94548, # mM/s
            'V_max_oxphos': 1.49397, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 0.3156, # unitless, # original
            # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            'r_rec': 0.045887e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 3, # Scaling factor for metabolic rates at rest   


            # Values from Barclay and Weber 2004
            'F_0': 0, # N, 
            'l_0': 11e-3, # m, 
            'mass': 4.1e-3, # g, 

            # Barclay and Weber 2004 experimental setup parameters 
            'velo_short': 1.3, # l0/s, Barclay and Weber 2004
            'freq': 80, # Hz, Frequency of stimulation 
            'max_dl': 0.1, # mm, Maximum length change

            # Activation model parameters 
            "Tau_1": 0.0422, # Assume constant value from MCL (2023)
            # "Tau_2": 0.125, # Scaling based on MCL (2023)
            "Tau_2": 0.057, #  BH 2012
            "K": 0.1025,
            "n": 3, # Hill coefficient for act mdoel
            
            # Mechanical parameters 
            'dedt_ce_max': 5, 
            'kappa': 0.18,

            # Initial energetics model 
            'r_cxb': 0.3786, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cat': 0.0662, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_sl': 0.239, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

        }, 
        'EDL': { 
            'c_c_tot': 29.5, # mM, Kushmerick et al. 1992 
            'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992 
            'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992 

            # Values to match recovery rate during initial contractoin
            # 'V_max_oxphos': 1.75, # mM/s
            # 'K_adp': 0.0615, # mM,
            # 'nh': 0.873, # unitless, 
            # 'r_rec': 2.41e5, # J / mol

            # Adjusted to match time course
            # 'V_max_oxphos': 3, # mM/s
            # 'K_adp': 0.0615, # mM,
            # 'nh': 0.873, # unitless, 
            # 'r_rec': 0.25 * 2.41e5, # J / mol

            #__________
            # SOL VALUES WITH SCLAING Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, BUGFIXED!
            # 'V_max_oxphos': 0.94548, # mM/s
            'V_max_oxphos': 0.5 * 1.49397, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 0.3156, # unitless, # original
            # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            'r_rec': 0.045887e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 3, # Scaling factor for metabolic rates at rest         

            'F_0': 0, # N, 
            'l_0': 8.9e-3, # m,
            'mass': 3.9e-3, # g, 

            # Barclay and Weber 2004 experimental setup parameters 
            'velo_short': 2.8, # l0/s, Barclay and Weber 2004
            'freq': 160, # Hz, Frequency of stimulation, BW2004
            # 'freq': 250, # Hz, Frequency of stimulation, Adjusted for tetenanus
            'max_dl': 0.2, # mm, Maximum length change

            # Activation model parameters 
            "Tau_1": 0.0422, # Very little change between fibre type - assume constant (BH, 2003)
            # "Tau_2": 0.125/2, # Decay constant for fast twitch Fibres assuming 1/2 rate (Baylor and Hollingworth, 2003)
            "Tau_2": 0.011, # BH 2012
            "K": 0.1025,
            "n": 3, # Hill coefficient for activation model

            # Mechanical parameters 
            'dedt_ce_max': 10, 
            'kappa': 0.29,

            # Energetics model 
            'r_cxb': 2.439, # F0l0/s, Maximum heat rate of isometric contraction (fast-type fibre)
            'r_cat': 0.1497, # F0l0/s, Maximum heat rate of isometric contraction (fast-type fibre)
            'r_sl': 1.0146, # W/F_0/l_0, Maximum shortening heat rate (fast-type fibre)

        },

        # Assume constant across all species and muscle fibre-types
        'V_ck_f': 100,# 100, # mM/s, Kushmerick 1998
        'K_b': 1.11, #mM, MacFarland 1994
        'K_ia': 0.135, # mM, MacFarland 1994
        'K_eq': 1.77e2, # ?, Assuming a pH of 7, Lawson 1979
        'K_iq': 3.5, # mM, MacFarland 1994
        'K_ib': 3.9, # mM, MacFarland 1994
        'K_p': 3.8, # mM, MacFarland 1994

        'Gatp': 60e3, # J/mol, Free energy of ATP (Barclay 2019)

        'k_see': 0, # Unused

    
}

'''
Compute necessary parameters from data 
'''

for muscle in ('SOL', 'EDL'):
    # Maximum isometric force (Assuming fixed max_iso_stress for now)    
    params[muscle]['F_0'] = params[muscle]['mass'] / params['rho0'] / params[muscle]['l_0'] * params['max_iso_stress']
    print(f'{muscle}: Maximum isometric stress: {params[muscle]["F_0"]}')

'''
Setup the simulation 
'''
def f_stim_length(t, params): 
    # Function to compute the length changes in the muscle 
    # returns both simulation times and lengths 

    cycle_length = params['cycle_length'] # s, Set at 0.3s for now (as in figure 1)
    t_cycle = t % cycle_length # Get the time with respect to the cycle
    N_cycles = params['N_cycles'] # Number of cycles to simulate

    # # time parameters (with respect to cycle time )
    # t_stim_start = 0 
    # t_stim_end = 0.1 * cycle_length
    # t_short_start = 0.05 * cycle_length
    # t_length_start = 0.15 * cycle_length
    # t_length_end =  cycle_length # Assume return to initial lenght by the end of the cycle
    # Time parameters edited to have a fixed stimulation time 
    if params['muscle'] == 'EDL':  
        t_stim_start = 0 
        t_stim_end = 0.063 # From paper 
        # t_short_start = 0.03
        t_short_start = 0.005
        t_length_start = 0.15
        t_length_end =  cycle_length # Assume return to initial lenght by the end of the cycle
    elif params['muscle'] == 'SOL': 
        t_stim_start = 0 
        t_stim_end = 0.125 # From paper
        t_short_start = 0.03
        t_length_start = 0.24
        t_length_end =  cycle_length # Assume return to initial lenght by the end of the cycle

    # Get the optimal length of the muscle
    l_0 = params[params['muscle']]['l_0']

    # OPT: Linear implementation
    # Extract necessary parameter values & compute velocities
    # Option 1:
    # Fix the maximum length across frequencies
    # dl_max = params[params['muscle']]['max_dl']  # Maximum length change
    # v_short = dl_max * l_0 / (t_length_start  - t_short_start) # shortening velocity
    # v_length = dl_max * l_0 / (t_length_end - t_length_start) # lengthening velocity
    # print(f'v_short = {v_short / l_0} l0/s')

    # Option 2: 
    # Fix the shortening rate across conditions
    v_short = -params[params['muscle']]['velo_short'] * l_0
    dl_max =  v_short / (t_length_start - t_short_start)
    v_length = (- v_short * (t_length_start - t_short_start)) / (t_length_end - t_length_start) # lengthening velocity
    print(f'v_short = {v_short / l_0} l0/s')
    print(f'dl_max = {dl_max / l_0} l0')
    print(f'v_length = {v_length / l_0} l0/s')

    # Change in length (mm)
    dl = ((t_cycle > t_short_start) * (t_cycle < t_length_start) * (v_short * (t_cycle - t_short_start))\
            + (t_cycle >= t_length_start) * (t_cycle < t_length_end) * (v_short * (t_length_start - t_short_start) + v_length * (t_cycle - t_length_start)))\
            * (t < cycle_length * N_cycles)

    # Toggle whether in stimulation or not (does not define frequency of stim here)
    stim = ((t_cycle >= t_stim_start) * (t_cycle <= t_stim_end) ) * (t < cycle_length * N_cycles)

    # Compute the stimulation times 
    # stim_times: vector (same shape as t) with 1 where a stimulus (spike) occurs, 0 otherwise
    stim_times = np.zeros_like(t, dtype=int)

    # determine stimulation frequency from params for the active muscle (fallback to 0)
    freq = params[params['muscle']]['freq']

    if freq > 0:
        period = 1.0 / freq

        # Get times when there is a stimulus 
        t_stim_period = t[stim] 
        # print(t_stim_period)

        # Get the firing times (accumulate into a Python list then convert)
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

        # optional debug prints
        # print('spike times:', t_fire_vec)
        # print('stim_times vector sum:', stim_times.sum())
            
    return stim, stim_times, dl

# Plot to verify conditions 
t_vec = np.linspace(params['t_start'], params['t_end'], int(10000 * params['t_end'])) 

freq_list = (0.5, 1, 2, 3, 4) # Hz, Frequencies for the cycles 
# freq_list = (1,2) # Hz, Frequencies for the cycles 

# Define a list of q10 temperature values to scale the rate of oxidative phosphorylation 
q10_oxphos_list = np.array((1, 1.25, 1.5))

component_names = ('q_a', 'q_m', 'q_sl', 'w', 'q_r')
component_colors = ('#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e')

# Storage for cross-muscle comparison
peak_qr_by_muscle = {}
tau_by_muscle = {}

# Loop over muscles
for muscle_name in ('SOL', 'EDL'):
    print(f'\n========== {muscle_name} ==========')
    params['muscle'] = muscle_name
    
    # INitialise plots 
    fig_energy, ax_energy = plt.subplots(layout = 'constrained')
    fig_tau, ax_tau = plt.subplots(layout = 'constrained')

    component_energy_abs = []
    efficiency_rows = []

    peak_qr_by_muscle[muscle_name] = np.zeros((len(q10_oxphos_list), len(freq_list)))
    tau_by_muscle[muscle_name] = np.zeros((len(q10_oxphos_list), len(freq_list)))

    # Deep copy the original parameter valuee 
    V_oxphos_max_val = copy.deepcopy(params[muscle]['V_max_oxphos'] )


    # Loop over Q10 values 
    for idx_q10, q10_oxphos in enumerate(q10_oxphos_list): 
        # peak_qr_vs_freq = []
        # tau_vs_freq = []


        for idx, freq in enumerate(freq_list): 
            stim_length = 1/freq # s, stimulation time 

            
            params['cycle_length'] = stim_length
            # params['N_cycles'] = np.floor(10 * freq)  # Choose to give the same length of time (ie. no change in frequencies)
            params['N_cycles'] = 10 # Fixed number of cycles (matches experimental conditions)
            stim_vec, stim_times_vec,  dl_vec = f_stim_length(t_vec, params)

            ''' 
            Simulate Ca2+ and mechanics 
            '''

            # Ca dynamics
            from Models.MUActivationModel import ActivationModel
            act_model = ActivationModel(params[params['muscle']], t_vec, True)
            idx_stims = np.nonzero(stim_times_vec)[0]
            stim_vec, ca_vec, catn_vec = act_model.runExcAct(idx_stims)

            # Mechanics 
            from Models.MechanicsModelSimple import MechModel 
            muscle = params['muscle']
            mech_model = MechModel(params[muscle]['l_0'], params[muscle]['dedt_ce_max'], params[muscle]['kappa'],params['k_see'])
            # Compute the strain and strain rates in the muscle 
            e_ce = dl_vec / params[muscle]['l_0'] + 0.1 # Get the strain adjusted so length change is over plateau
            dedt_ce = np.diff(e_ce, prepend = 0) / np.diff(t_vec, prepend = 1)

            # Plot one cycle for the current frequency condition.
            cycle_mask = (t_vec >= 0) * (t_vec < params['cycle_length'])
            t_cycle = t_vec[cycle_mask]
            e_ce_cycle = e_ce[cycle_mask]
            dedt_ce_cycle = dedt_ce[cycle_mask]

            # Compute the force directly  
            force_direct =  mech_model.computeForce(catn_vec, e_ce + 1, dedt_ce)

            # Compute the initial energetics 
            from Models.MUEnergeticsModelSimple_SplitVars import EnergeticsModel
            energy_model = EnergeticsModel()
            q_a, q_m, q_sl, w = energy_model.actEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force_direct, mech_model)
            E_tot = q_a + q_m + q_sl + w  # F0l0/s, Total energy 

            # Convert units to input for bioenergetics model 
            E_initial_converted = E_tot * params[muscle]['F_0'] * params[muscle]['l_0'] / params[muscle]['mass'] # W/g

            # Update the phi_oxphos_max with new values 
            params[muscle]['V_max_oxphos'] = q10_oxphos * V_oxphos_max_val

            # Run bioenergetics
            from Models.BioenergeticsSimple import Bioenergetics
            bioenergetic_model = Bioenergetics(params) 
            t_span = (t_vec[0], t_vec[-1]) 
            c_atp_0 = params[muscle]['c_atp_0']
            # Solve the model
            sol = bioenergetic_model.solveBioenergetics(t_span, c_atp_0, t_vec, E_initial_converted)
            # Compute the energetic rates 
            scale =  params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0'] 
            q_r = bioenergetic_model.computeRecoveryEnergetics(sol.t, sol.y[0,]) * scale # F0l0/s = J/g/s * g/F0l0

            # Compute the scaler to get correct units
            energy_unit_scaler = params[muscle]['F_0'] * params[muscle]['l_0'] / params[muscle]['mass'] * 1e3 # convert from F0l0/s to mW/g 

            # Plot the total energy over the cycle
            ax_energy.plot(t_vec, cumtrapz(E_tot, t_vec, initial = 0) * energy_unit_scaler, color = palette[idx_q10], alpha = 0.25, label = None) 
            ax_energy.plot(t_vec, cumtrapz(q_r, t_vec, initial = 0) * energy_unit_scaler, color = palette[idx_q10], ls = ':', alpha = 0.5, label = None) 
            ax_energy.plot(t_vec, cumtrapz(E_tot + q_r, t_vec, initial = 0) * energy_unit_scaler, color = palette[idx_q10], label =  f'q10 = {q10_oxphos}') 
            ax_energy.set_xlabel('Time (s)')
            ax_energy.set_ylabel('Energy  ($mJ g^{-1}$)')

            # Store absolute end-of-trial energies for component contribution bar charts.
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

            ax_tau.plot(t_rel, y_decay, color = palette[idx_q10], alpha = 0.35, label = None)
            ax_tau.plot(t_rel, exp_decay(t_rel, *popt), '--', color = palette[idx_q10], label = f'q10 = {q10_oxphos}, f = {freq} Hz fit ($\\tau$ = {tau_fit:.2f} s)')

            # Compute the peak recovery rate 
            peak_qr_by_muscle[muscle_name][idx_q10, idx] = np.max(q_r[mask] * energy_unit_scaler)
            tau_by_muscle[muscle_name][idx_q10, idx] = tau_fit

            # Compute efficiencies from integrated energies over the full simulation window.
            E_tot_end = cumtrapz(E_tot, t_vec, initial = 0)[-1]
            E_rec_end = cumtrapz(q_r, t_vec, initial = 0)[-1]
            W_end = cumtrapz(w, t_vec, initial = 0)[-1]

            eta_init = W_end / E_tot_end if np.abs(E_tot_end) > 0 else np.nan
            eta_total = W_end / (E_tot_end + E_rec_end) if np.abs(E_tot_end + E_rec_end) > 0 else np.nan
            efficiency_ratio = (eta_init / eta_total - 1) if (not np.isnan(eta_total) and eta_total > 0) else np.nan
            efficiency_rows.append((q10_oxphos, freq, eta_init, eta_total, efficiency_ratio))

            print(f'Fitted time constant (tau) = {tau_fit:.3f} s')

        # peak_qr_by_muscle[muscle_name][idx_q10] = np.array(peak_qr_vs_freq)
        # tau_by_muscle[muscle_name][idx_q10] = np.array(tau_vs_freq)

    ax_tau.set_xlabel('Time since end of stimulation (s)')
    ax_tau.set_ylabel('Total energy rate ($mW g^{-1}$)')
    ax_tau.set_title(f'{muscle_name} - Recovery Time Constant')
    ax_tau.legend(loc = 'upper right')

    ax_energy.legend(loc = 'upper right')

    # # Plot component contributions vs frequency (absolute energies at trial end)
    # component_energy_abs = np.array(component_energy_abs)
    # x = np.arange(len(freq_list))
    # n_comp = len(component_names)
    # bar_width = 0.15

    # fig_comp_abs, ax_comp_abs = plt.subplots(layout = 'constrained')
    # for comp_idx, comp_name in enumerate(component_names):
    #     offset = (comp_idx - (n_comp - 1) / 2) * bar_width
    #     ax_comp_abs.bar(
    #         x + offset,
    #         component_energy_abs[:, comp_idx],
    #         width = bar_width,
    #         color = component_colors[comp_idx],
    #         label = comp_name
    #     )
    # ax_comp_abs.set_title(f'{muscle_name} - Absolute Contributions')
    # ax_comp_abs.set_xticks(x)
    # ax_comp_abs.set_xticklabels([f'{freq} Hz' for freq in freq_list])
    # ax_comp_abs.set_xlabel('Cycle frequency')
    # ax_comp_abs.set_ylabel('End-of-trial energy ($mJ g^{-1}$)')
    # ax_comp_abs.legend()

    # # Plot relative component contributions (normalised by total end-of-trial energy)
    # row_totals = np.sum(component_energy_abs, axis = 1)
    # component_energy_rel = component_energy_abs / row_totals[:, None]

    # fig_comp_rel, ax_comp_rel = plt.subplots(layout = 'constrained')
    # bar_width_rel = 0.35
    # x_qi = x - bar_width_rel / 2
    # x_qr = x + bar_width_rel / 2

    # # Stacked q_i bar for each condition: q_a + q_m + q_sl + w
    # q_i_bottom = np.zeros_like(x, dtype=float)
    # for comp_idx in range(4):
    #     ax_comp_rel.bar(
    #         x_qi,
    #         component_energy_rel[:, comp_idx],
    #         width = bar_width_rel,
    #         bottom=q_i_bottom,
    #         color=component_colors[comp_idx],
    #         label=component_names[comp_idx]
    #     )
    #     q_i_bottom += component_energy_rel[:, comp_idx]

    # # Separate q_r bar for each condition
    # ax_comp_rel.bar(
    #     x_qr,
    #     component_energy_rel[:, 4],
    #     width=bar_width_rel,
    #     color=component_colors[4],
    #     label=component_names[4]
    # )
    # ax_comp_rel.set_title(f'{muscle_name} - Relative Contributions')
    # ax_comp_rel.set_xticks(x)
    # ax_comp_rel.set_xticklabels([f'{freq} Hz' for freq in freq_list])
    # ax_comp_rel.set_xlabel('Cycle frequency')
    # ax_comp_rel.set_ylabel('Relative contribution at trial end')
    # ax_comp_rel.legend()



    # Print efficiency table in terminal for the current muscle.
    print(f'\n{muscle_name} efficiency table')
    print('q10_oxphos | freq_Hz | eta_init = W/E_tot | eta_total = W/(E_tot + E_rec) | E_rec / E_tot')
    print('----------+---------+---------------------+-------------------------------+---------------')
    for q10_oxphos, freq, eta_init, eta_total, efficiency_ratio in efficiency_rows:
        print(f'{q10_oxphos:10.2f} | {freq:7.2f} | {eta_init:19.6f} | {eta_total:29.6f} | {efficiency_ratio:14.6f}')

    efficiency_array = np.array(efficiency_rows, dtype=float)
    mean_vals = np.nanmean(efficiency_array, axis=0)
    sem_vals = np.nanstd(efficiency_array, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(efficiency_array), axis=0))
    print(' mean±SEM | '
        f'{mean_vals[0]:.6f} ± {sem_vals[0]:.6f} | '
        f'{mean_vals[1]:.6f} ± {sem_vals[1]:.6f} | '
        f'{mean_vals[2]:.6f} ± {sem_vals[2]:.6f} | '
        f'{mean_vals[3]:.6f} ± {sem_vals[3]:.6f} | '
        f'{mean_vals[4]:.6f} ± {sem_vals[4]:.6f}')

# Plot peak recovery rate versus frequency for both muscles
fig_peak_qr_compare, ax_peak_qr_compare = plt.subplots(layout = 'constrained')
for idx_q10, q10_oxphos in enumerate(q10_oxphos_list):
    ax_peak_qr_compare.plot(
        freq_list,
        peak_qr_by_muscle['SOL'][idx_q10],
        '-o',
        color=palette[idx_q10],
        alpha=0.9,
        label=f'SOL, q10 = {q10_oxphos}'
    )
    ax_peak_qr_compare.plot(
        freq_list,
        peak_qr_by_muscle['EDL'][idx_q10],
        '--s',
        color=palette[idx_q10],
        alpha=0.9,
        label=f'EDL, q10 = {q10_oxphos}'
    )
ax_peak_qr_compare.set_xlabel('Cycle frequency (Hz)')
ax_peak_qr_compare.set_ylabel('Peak recovery rate ($mW g^{-1}$)')
ax_peak_qr_compare.grid(True, alpha = 0.3)

# Overlay BW2004 experimental data (black, legend label: EXP).
exp_rrecmax_sol = np.genfromtxt('Data/BW2004_data_rrecmax_SOL.csv', delimiter=',', names=True)
exp_rrecmax_edl = np.genfromtxt('Data/BW2004_data_rrecmax_EDL.csv', delimiter=',', names=True)
ax_peak_qr_compare.plot(exp_rrecmax_sol['freq'][0:len(freq_list)], exp_rrecmax_sol['rrecmax'][0:len(freq_list)], 'k-o', lw=1.5, ms=4, label='EXP')
ax_peak_qr_compare.plot(exp_rrecmax_edl['freq'][0:len(freq_list)], exp_rrecmax_edl['rrecmax'][0:len(freq_list)], 'k--s', lw=1.5, ms=4, label='_nolegend_')

ax_peak_qr_compare.legend(loc = 'upper right')
fig_peak_qr_compare.savefig('Figures/BW_q10sens_rrecmax.jpg')

# Plot fitted time constants versus frequency for both muscles
fig_tau_freq_compare, ax_tau_freq_compare = plt.subplots(layout = 'constrained')
for idx_q10, q10_oxphos in enumerate(q10_oxphos_list):
    ax_tau_freq_compare.plot(
        freq_list,
        tau_by_muscle['SOL'][idx_q10],
        '-o',
        color=palette[idx_q10],
        alpha=0.9,
        label=f'SOL, q10 = {q10_oxphos}'
    )
    ax_tau_freq_compare.plot(
        freq_list,
        tau_by_muscle['EDL'][idx_q10],
        '--s',
        color=palette[idx_q10],
        alpha=0.9,
        label=f'EDL, q10 = {q10_oxphos}'
    )
ax_tau_freq_compare.set_xlabel('Cycle frequency (Hz)')
ax_tau_freq_compare.set_ylabel('Time constant $\\tau$ (s)')
ax_tau_freq_compare.grid(True, alpha = 0.3)

exp_tau_sol = np.genfromtxt('Data/BW2004_data_tau_SOL.csv', delimiter=',', names=True)
exp_tau_edl = np.genfromtxt('Data/BW2004_data_tau_EDL.csv', delimiter=',', names=True)
ax_tau_freq_compare.plot(exp_tau_sol['freq'][0:len(freq_list)], exp_tau_sol['tau'][0:len(freq_list)], 'k-o', lw=1.5, ms=4, label='EXP')
ax_tau_freq_compare.plot(exp_tau_edl['freq'][0:len(freq_list)], exp_tau_edl['tau'][0:len(freq_list)], 'k--s', lw=1.5, ms=4, label='_nolegend_')

ax_tau_freq_compare.legend(loc = 'upper right')
fig_tau_freq_compare.savefig('Figures/BW_q10sens_tau.jpg')


plt.show()
