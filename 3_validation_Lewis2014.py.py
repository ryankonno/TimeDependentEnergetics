'''
This code implements the model as designed in Kushmerick 1999, and adapted in Vicini 2000

This version of the code has been adapted such that the initial resting rate of the model will altered based of Vmax. This allows for a fixed c_pcr_0 while othere parameters are varied... I do not believe it is expected that if rate constants vary then there shoulld be an alteratation in steady-state metabolism

This code fits the recovery heat parameter + ATP Pcr mdoel parameters to the Barclay 1995 dataset

Adapted to plot the model results from teh Barclay 2004 experiments with varying frequencies

Adapted to use separate variables for the cat and cxb heat terms.

Dual-muscle version looping over SOL and EDL with cross-muscle comparison plots.

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

            'max_iso_stress': 2.37e5, # N/m^2, B1996

            # #__________
            # # Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, BUGFIXED!
            # # 'V_max_oxphos': 0.94548, # mM/s
            # 'V_max_oxphos': 1.49397, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            # 'K_adp': 0.058, # mM,
            # 'nh': 0.3156, # unitless, # original
            # # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            # 'r_rec': 0.045887e6, # J / mol, Obtained from efficiency calculation 
            # 'gamma': 3, # Scaling factor for metabolic rates at rest   
            #__________
            # # Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, BUGFIXED!
            # 'V_max_oxphos': 0.94548, # mM/s
            'V_max_oxphos': 2 * 1.9322, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 0.61325, # unitless, # original
            # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            'r_rec': 0.5 * 0.16730e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 1, # Scaling factor for metabolic rates at rest    


            # Values from Barclay and Weber 2004
            'F_0': 0, # N, 
            'l_0': 11e-3, # m, 
            'mass': 4.1e-3, # g, 

            # Barclay and Weber 2004 experimental setup parameters 
            'velo_short': 1.3, # l0/s, Barclay and Weber 2004
            # 'freq': 80, # Hz, Frequency of stimulation 
            'freq': 200, # Hz, Frequency of stimulation 
            'max_dl': 0.1, # mm, Maximum length change

            # Activation model parameters 
            'Tau_1': 0.038,  # requested
            'Tau_2': 0.055,  # B2012 30deg
            "K": 0.25,
            "n": 1.99, # Hill coefficient for act mdoel

            
            # Mechanical parameters 
            'dedt_ce_max': 6, 
            'kappa': 0.18,

            # # Initial energetics model 
            # 'r_cxb':  0.42406, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_cat': 0.04845, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_sl':  0.26774, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)
            # # Optimisation with a submax scaling factor, Cat NO scaling
            # 'r_cxb':  0.40197, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_cat': 0.0479003, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'cxb_scale':  0.566683, # unitless, cxb scale factor
            # 'r_sl':  0.26774, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)
            # Optimisation with a submax scaling factor, Cat NO scaling, B2010 data
            'r_cxb':  0.2473242, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cat': 0.029479, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'cxb_scale':  0.5665, # unitless, cxb scale factor
            'r_sl':  0.26774, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

            # Konno et al., 2025 model parameters 
            'r1': 0.6177,
            'r2': 0.2342,

        }, 
        'EDL': { 
            'c_c_tot': 29.5, # mM, Kushmerick et al. 1992 
            'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992 
            'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992 

            'max_iso_stress': 3.01e5, # N/m^2, B1996
            # 'max_iso_stress': 2.5e5, # N/m^2, B1996
            

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

            # #__________
            # # SOL VALUES WITH SCLAING Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, BUGFIXED!
            # # 'V_max_oxphos': 0.94548, # mM/s
            # 'V_max_oxphos': 2 * 1.49397, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            # 'K_adp': 0.058, # mM,
            # 'nh': 0.3156, # unitless, # original
            # # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            # 'r_rec': 0.045887e6, # J / mol, Obtained from efficiency calculation 
            # 'gamma': 3, # Scaling factor for metabolic rates at rest         
            #__________
            # # Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, BUGFIXED!
            # 'V_max_oxphos': 0.94548, # mM/s
            'V_max_oxphos': 2 * 1.9322, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 0.61325, # unitless, # original
            # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            'r_rec': 0.5 * 0.16730e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 1, # Scaling factor for metabolic rates at rest    

            'F_0': 0, # N, 
            'l_0': 8.9e-3, # m,
            'mass': 3.9e-3, # g, 

            # Barclay and Weber 2004 experimental setup parameters 
            'velo_short': 2.8, # l0/s, Barclay and Weber 2004
            'freq': 160, # Hz, Frequency of stimulation, BW2004
            # 'freq': 250, # Hz, Frequency of stimulation, Adjusted for tetenanus
            'max_dl': 0.2, # mm, Maximum length change

            # Activation model parameters 
            'Tau_1': 0.011,  # requested
            'Tau_2': 0.011,  # BH 2003, fibre bundle data
            "K": 0.45,
            "n": 2.89, # Hill coefficient for activation model

            # Mechanical parameters 
            'dedt_ce_max': 15, 
            'kappa': 0.29,

            # # Energetics model 
            # 'r_cxb': 1.86285, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_cat': 0.320083, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_sl':  0.77495, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)
            # # Optimisation with a submax scaling factor , Cat NO scaling
            # 'r_cxb':  1.8131448, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_cat': 0.04779, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'cxb_scale':  0.2536262, # unitless, cxb scale factor
            # 'r_sl':  0.26774, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)
            # Optimisation with a submax scaling factor , Cat NO scaling, B2010 FIT
            'r_cxb':  0.76267727, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cat': 0.01992, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'cxb_scale':  0.2565930, # unitless, cxb scale factor
            'r_sl':  0.697, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

            # Konno et al., 2025 model parameters 
            'r1': 2.7919,
            'r2': 0.697,


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
    # Maximum isometric force
    params[muscle]['F_0'] = (
        params[muscle]['mass'] / params['rho0'] /
        params[muscle]['l_0'] *
        params[muscle]['max_iso_stress']
    )
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
        t_short_start = 0.01 # Choose to get reasonable power & work... (sampe proportion as soleus values)
        # t_short_start = 0.005
        t_length_start = 0.15
        t_length_end =  cycle_length # Assume return to initial lenght by the end of the cycle
    elif params['muscle'] == 'SOL': 
        t_stim_start = 0 
        t_stim_end = 0.125 # From paper
        t_short_start = 0.05
        t_length_start = 0.24
        t_length_end =  cycle_length # Assume return to initial lenght by the end of the cycle

    # Get the optimal length of the muscle
    l_0 = params[params['muscle']]['l_0']


    # Fix the shortening rate across conditions
    v_short = -params[params['muscle']]['velo_short'] * l_0
    dl_max =  v_short / (t_length_start - t_short_start)
    v_length = (
        (- v_short * (t_length_start - t_short_start)) /
        (t_length_end - t_length_start)
    )  # lengthening velocity

    # Change in length (mm)
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

    # determine stimulation frequency from params for the active muscle (fallback to 0)
    freq = params[params['muscle']]['freq']

    if freq > 0:
        period = 1.0 / freq

        # Get times when there is a stimulus 
        t_stim_period = t[stim] 

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
            
    return stim, stim_times, dl

# Plot to verify conditions 
t_vec = np.linspace(params['t_start'], params['t_end'], int(10000 * params['t_end'])) 

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

# Loop over muscles
for muscle_name in ('SOL', 'EDL'):
    print(f'\n========== {muscle_name} ==========')
    params['muscle'] = muscle_name
    
    # Initialise plots 
    fig_energy, ax_energy = plt.subplots(layout = 'constrained')
    fig_tau, ax_tau = plt.subplots(layout = 'constrained')
    fig_strain_cycle, axs_strain_cycle = plt.subplots(2, 1, layout='constrained', sharex=False)
    fig_force_cycle, ax_force_cycle = plt.subplots(layout='constrained')
    
    # Figure for time-varying energy components during one cycle
    fig_energy_components, ax_energy_comp = plt.subplots(
        layout='constrained', figsize=(10, 6)
    )

    component_energy_abs = []
    peak_qr_vs_freq = []
    tau_vs_freq = []
    efficiency_rows = []
    total_energy_out_f0l0_vs_freq = []
    total_energy_out_mJg_vs_freq = []

    for idx, stim_freq in enumerate(stim_freq_list): 
        cycle_length = 1.0 / cycle_freq_hz
        print(f'Stimulation frequency: {stim_freq} Hz')

        
        params['cycle_length'] = cycle_length
        params[muscle_name]['freq'] = stim_freq
        params['N_cycles'] = 10 # Fixed number of cycles (matches experimental conditions)
        stim_vec, stim_times_vec,  dl_vec = f_stim_length(t_vec, params)

        ''' 
        Simulate Ca2+ and mechanics 
        '''

        # Ca dynamics
        from Models.MUActivationModel import ActivationModel
        act_model = ActivationModel(params[params['muscle']], t_vec, True)
        idx_stims = np.nonzero(stim_times_vec)[0]
        stim_vec, ca_vec, catn_vec = act_model.runExcAct(idx_stims, w_0 = 0.004)

        # Mechanics 
        from Models.MechanicsModelSimple import MechModel 
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
        axs_strain_cycle[0].plot(t_cycle, e_ce_cycle, color=palette[idx], label=f'{stim_freq} Hz')
        axs_strain_cycle[1].plot(t_cycle, dedt_ce_cycle, color=palette[idx], label=f'{stim_freq} Hz')

        # Plot the force
        # Compute the force directly  
        force_direct =  mech_model.computeForce(catn_vec, e_ce + 1, dedt_ce)
        force_cycle = force_direct[cycle_mask]
        ax_force_cycle.plot(t_cycle, force_cycle, color=palette[idx], label=f'{stim_freq} Hz')

        # Compute the initial energetics 
        from Models.MUEnergeticsModelSimple_SplitVars import EnergeticsModel
        energy_model = EnergeticsModel()
        q_a, q_m, q_sl, w = energy_model.actEnergetics(
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
        from Models.BioenergeticsSimple import Bioenergetics
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
                color=component_colors[0], linewidth=2
            )
            ax_energy_comp.plot(
                t_cycle, q_m_cum, label='$q_m$ (maintenance)',
                color=component_colors[1], linewidth=2
            )
            ax_energy_comp.plot(
                t_cycle, q_sl_cum, label='$q_{sl}$ (shortening)',
                color=component_colors[2], linewidth=2
            )
            ax_energy_comp.plot(
                t_cycle, w_cum, label='$w$ (mechanical work)',
                color=component_colors[3], linewidth=2
            )
            ax_energy_comp.plot(
                t_cycle, q_r_cum, label='$q_r$ (recovery)',
                color=component_colors[4], linewidth=2
            )
            
            # Add labels and formatting
            ax_energy_comp.set_xlabel('Time (s)', fontsize=12)
            ax_energy_comp.set_ylabel('Cumulative Energy (mJ/g)', fontsize=12)
            ax_energy_comp.legend(loc='upper left', fontsize=11)
            ax_energy_comp.grid(True, alpha=0.3)
            fig_energy_components.suptitle(
                f'{muscle_name}: Time-varying energy components '\
                f'({stim_freq_list[0]} Hz)',
                fontsize=12, fontweight='bold'
            )

        # Plot total energy
        ax_energy.plot(
            t_vec,
            cumtrapz(E_tot, t_vec, initial=0) *
            energy_unit_scaler,
            label='$e_{init}$', color=palette[idx], alpha=0.25
        )
        ax_energy.plot(
            t_vec,
            cumtrapz(q_r, t_vec, initial=0) *
            energy_unit_scaler,
            label='$q_r$', color=palette[idx], ls=':',
            alpha=0.5
        )
        ax_energy.plot(
            t_vec,
            cumtrapz(E_tot + q_r, t_vec, initial=0) *
            energy_unit_scaler,
            label='$q_r + e_{init}$', color=palette[idx]
        ) 
        ax_energy.set_xlabel('Time (s)')
        ax_energy.set_ylabel('Energy  ($mJ g^{-1}$)')

        # Compute energetics using previous model
        energy_rate_data, energy_data_ = energy_model.dHdt(
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

        ax_tau.plot(t_rel, y_decay, color = palette[idx], alpha = 0.35, label = f'{stim_freq} Hz decay')
        ax_tau.plot(
            t_rel, exp_decay(t_rel, *popt), '--',
            color=palette[idx],
            label=f'{stim_freq} Hz fit ($\\tau$ = '
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

    ax_tau.set_xlabel('Time since end of stimulation (s)')
    ax_tau.set_ylabel('Total energy rate ($mW g^{-1}$)')
    ax_tau.set_title(f'{muscle_name} - Recovery Time Constant')
    ax_tau.legend(loc = 'upper right')

    axs_strain_cycle[0].set_title(f'{muscle_name} - One-Cycle Strain')
    axs_strain_cycle[0].set_xlabel('Time within cycle (s)')
    axs_strain_cycle[0].set_ylabel('$e_{ce}$')
    axs_strain_cycle[0].grid(True, alpha=0.3)
    # axs_strain_cycle[0].legend()
    axs_strain_cycle[1].set_xlabel('Time within cycle (s)')
    axs_strain_cycle[1].set_ylabel('$\dot e_{ce}$ ($s^{-1}$)')
    axs_strain_cycle[1].grid(True, alpha=0.3)
    # axs_strain_cycle[1].legend()

    ax_force_cycle.set_title(f'{muscle_name} - One-Cycle Force')
    ax_force_cycle.set_xlabel('Time within cycle (s)')
    ax_force_cycle.set_ylabel('Force (N)')
    ax_force_cycle.grid(True, alpha=0.3)
    ax_force_cycle.legend(loc = 'upper right')

    # Plot component contributions vs frequency (absolute energies at trial end)
    component_energy_abs = np.array(component_energy_abs)
    x = np.arange(len(stim_freq_list))
    n_comp = len(component_names)
    bar_width = 0.15

    fig_comp_abs, ax_comp_abs = plt.subplots(layout = 'constrained')
    for comp_idx, comp_name in enumerate(component_names):
        offset = (comp_idx - (n_comp - 1) / 2) * bar_width
        ax_comp_abs.bar(
            x + offset,
            component_energy_abs[:, comp_idx],
            width = bar_width,
            color = component_colors[comp_idx],
            label = comp_name
        )
    ax_comp_abs.set_title(f'{muscle_name} - Absolute Contributions')
    ax_comp_abs.set_xticks(x)
    ax_comp_abs.set_xticklabels([f'{stim_freq} Hz' for stim_freq in stim_freq_list])
    ax_comp_abs.set_xlabel('Stimulation frequency (Hz)')
    ax_comp_abs.set_ylabel('End-of-trial energy ($mJ g^{-1}$)')
    ax_comp_abs.legend(loc = 'upper right')

    # Plot relative component contributions (normalised by total end-of-trial energy)
    row_totals = np.sum(component_energy_abs, axis = 1)
    component_energy_rel = component_energy_abs / row_totals[:, None]

    fig_comp_rel, ax_comp_rel = plt.subplots(layout = 'constrained')
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
        
    # # Save the energetics figure 
    # fig_energy.savefig('Figures/B2004_energy_' + muscle + '.jpg')
    # fig_energy_components.savefig(
    #     'Figures/B2004_energy_components_' + muscle + '.jpg'
    # )

# Plot peak recovery rate versus frequency for both muscles
fig_peak_qr_compare, ax_peak_qr_compare = plt.subplots(layout = 'constrained')
ax_peak_qr_compare.plot(stim_freq_list, peak_qr_by_muscle['SOL'], '-o', label='SOL', color='#1f77b4')
ax_peak_qr_compare.plot(stim_freq_list, peak_qr_by_muscle['EDL'], '-o', label='EDL', color="#d62728")
ax_peak_qr_compare.set_xlabel('Stimulation frequency (Hz)')
ax_peak_qr_compare.set_ylabel('Peak recovery rate ($mW g^{-1}$)')

exp_rrecmax_sol = np.genfromtxt('Data/BW2004_data_rrecmax_SOL.csv', delimiter=',', names=True)
exp_rrecmax_edl = np.genfromtxt('Data/BW2004_data_rrecmax_EDL.csv', delimiter=',', names=True)
ax_peak_qr_compare.plot(
    exp_rrecmax_sol['freq'][0:len(stim_freq_list)],
    exp_rrecmax_sol['rrecmax'][0:len(stim_freq_list)],
    'k-o', lw=1.5, ms=4, label='EXP'
)
ax_peak_qr_compare.plot(
    exp_rrecmax_edl['freq'][0:len(stim_freq_list)],
    exp_rrecmax_edl['rrecmax'][0:len(stim_freq_list)],
    'k--s', lw=1.5, ms=4, label='_nolegend_'
)

ax_peak_qr_compare.grid(True, alpha = 0.3)
ax_peak_qr_compare.legend(loc = 'upper right')
# fig_peak_qr_compare.savefig('Figures/B2004_SepVars_rrecmax_comp.jpg')

# Plot initial and total energy-rate output against power output for both muscles.
fig_energy_power_compare, ax_energy_power_compare = plt.subplots(layout = 'constrained')
ax_energy_power_compare.plot(
    power_output_by_muscle['SOL'],
    initial_energy_rate_output_by_muscle['SOL'],
    '--o', label='SOL initial rate', color='#1f77b4'
)
ax_energy_power_compare.plot(
    power_output_by_muscle['SOL'],
    total_energy_rate_output_by_muscle['SOL'],
    '-o', label='SOL total rate', color='#1f77b4'
)
ax_energy_power_compare.plot(
    power_output_by_muscle['EDL'],
    initial_energy_rate_output_by_muscle['EDL'],
    '--o', label='EDL initial rate', color='#d62728'
)
ax_energy_power_compare.plot(
    power_output_by_muscle['EDL'],
    total_energy_rate_output_by_muscle['EDL'],
    '-o', label='EDL total rate', color='#d62728'
)
ax_energy_power_compare.set_xlabel('Power output ($mW g^{-1}$)')
ax_energy_power_compare.set_ylabel('Total energy rate output ($mW g^{-1}$)')
ax_energy_power_compare.grid(True, alpha = 0.3)
ax_energy_power_compare.legend(loc = 'upper right')
# fig_energy_power_compare.savefig('Figures/B2004_energy_vs_power_comp.jpg')

# Plot total energy output normalised to F_0 l_0 for both muscles.
fig_total_energy_compare, ax_total_energy_compare = plt.subplots(layout = 'constrained')
ax_total_energy_compare.plot(
    stim_freq_list, total_energy_output_by_muscle['SOL'],
    '-o', label='SOL', color='#1f77b4'
)
ax_total_energy_compare.plot(
    stim_freq_list, total_energy_output_by_muscle['EDL'],
    '-o', label='EDL', color='#d62728'
)
ax_total_energy_compare.set_xlabel('Stimulation frequency (Hz)')
ax_total_energy_compare.set_ylabel('Total energy ($F_0 l_0$)')
ax_total_energy_compare.grid(True, alpha = 0.3)
ax_total_energy_compare.legend(loc = 'upper right')
# fig_total_energy_compare.savefig('Figures/B2004_total_energy_F0l0_comp.jpg')

# Plot total energy output in mJ g^-1 for both muscles.
fig_total_energy_mJg_compare, ax_total_energy_mJg_compare = plt.subplots(layout = 'constrained')
ax_total_energy_mJg_compare.plot(
    stim_freq_list, total_energy_output_mJg_by_muscle['SOL'],
    '-o', label='SOL', color='#1f77b4'
)
ax_total_energy_mJg_compare.plot(
    stim_freq_list, total_energy_output_mJg_by_muscle['EDL'],
    '-o', label='EDL', color='#d62728'
)
ax_total_energy_mJg_compare.set_xlabel('Stimulation frequency (Hz)')
ax_total_energy_mJg_compare.set_ylabel('Total energy ($mJ g^{-1}$)')
ax_total_energy_mJg_compare.grid(True, alpha = 0.3)
ax_total_energy_mJg_compare.legend(loc = 'upper right')
# fig_total_energy_mJg_compare.savefig('Figures/B2004_total_energy_mJg_comp.jpg')

# Plot fitted time constants versus frequency for both muscles
fig_tau_freq_compare, ax_tau_freq_compare = plt.subplots(layout = 'constrained')
ax_tau_freq_compare.plot(stim_freq_list, tau_by_muscle['SOL'], '-o', label='SOL', color='#1f77b4')
ax_tau_freq_compare.plot(stim_freq_list, tau_by_muscle['EDL'], '-o', label='EDL', color='#d62728')
ax_tau_freq_compare.set_xlabel('Stimulation frequency (Hz)')
ax_tau_freq_compare.set_ylabel('Time constant $\\tau$ (s)')

exp_tau_sol = np.genfromtxt('Data/BW2004_data_tau_SOL.csv', delimiter=',', names=True)
exp_tau_edl = np.genfromtxt('Data/BW2004_data_tau_EDL.csv', delimiter=',', names=True)

ax_tau_freq_compare.grid(True, alpha = 0.3)
ax_tau_freq_compare.legend(loc = 'upper right')

# fig_tau_freq_compare.savefig('Figures/B2004_SepVars_tau_comp.jpg')

plt.show()
