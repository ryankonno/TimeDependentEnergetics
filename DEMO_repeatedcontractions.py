
'''
This code simulates the energetics for repeated isometric contractions. 

All parameters for the code are contained within the file. 

Within the parameter dictionary (params) are time variables to set up the simulation including the contraction frequency and stimulation frequency. Muscle specific parameter for the energetics model are also included for the SOL and EDL. 

To investigate the role of contraction frequency or stimulation frequency, the parameter sim_type can be chosen to choose the simulation setup. The options are 
    - varycontrfreq: vary the frequency of the contractions with stimulation frequency as defined
    - varystimfreq: vary the frequency of the stimulation with contraction frequency as defined 
    - single: Perfrom a single stimulation and contraction frequency with parameters as defined


The outputs from this code include two tables and two figures: 

    Table 1: Comparison of time constants and recovery to initial energetics across different frequencies 

    Table 2: Total energy be component over the simulation 

    Figure 1: Cumulative energy over the simulation including total energy (solid line), initial energy (dotted line), and recovery energy (dashed line)

    Figure 2: Total energy spent per componentent

    NOTE: for output, the frequency will correspond to either contraction frequency or stimulation frequency depending on the sim_type parameter

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
palette = ("#32cd9c", "#f67410", "#2b21b8", "#C21599", "#83d921", "#1ab6e9")
import sys 
sys.path.append('./')

from Models.ActivationModel import ActivationModel
from Models.MechanicsModel import MechModel 
from Models.InitialEnergeticsModel import EnergeticsModel
from Models.BioenergeticsModel import Bioenergetics

# Define parameters
# note: could also import parameters from parameters_muscle.py
params = {
    # Time parameters for setting up the protocol 
    't_start': 0, # s
    't_end': 50, # s, end time of the stimulation
    'stim_freq': 150, # Hz, Freqeucy of the stimulation 
    'contr_freq': 1, # Hz, Freqeucy of the repeated contractions 
    'N_cycles': 5, # Number of repeated contractions
    'duty_factor': 0.5, # Fraction of the cycle to apply the stimulation

    # Simulation type
    # Options: 
    # - varycontrfreq: vary the frequency of the contractions with stimulation frequency as defined above 
    # - varystimfreq: vary the frequency of the stimulation with contraction frequency as defined above 
    # - single: Perfrom a single stimulation and contraction frequency with parameters as defined above
    'sim_type': 'varycontrfreq', 
    # If varycontrfreq then the following parameters are relevant 
    'contr_freq_list': (0.5, 1, 2, 4),
    # If varystimfreq, then the following parameter are relevant 
    'stim_freq_list': (1, 2, 4, 8, 16, 32, 64, 128),

    # Define the muscle for the simulation 
    'muscle': 'SOL', # 'SOL' or 'EDL'

    # General muscle parameters
    'rho0':  1e6, # g/m^3, Density of muscle

        # Mouse data 
        'SOL': {

            # Excitation-activation parameters
            'Tau_1': 0.038,  # s, Close et al., 1967
            'Tau_2': 0.055,  # s, Close et al., 1967
            "K": 0.25, # mM, Concentration of ATP at 50% activation
            "n": 1.99, # Hill coefficient for act mdoel

            # Mechanical parameters 
            'F_0': 0, # N, Barclay and Weber 2004
            'l_0': 11e-3, # m, Barclay and Weber 2004
            'mass': 4.1e-3, # g, Barclay and Weber 2004
            'max_iso_stress': 2.37e5, # N/m^2, Barclay 1996
            'dedt_ce_max': 6,  # s^-1, Barclay 2010
            'kappa': 0.18,
            
            # Initial energetics model 
            'r_cxb':   0.247, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cat': 0.0295, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'cxb_scale':  0.567, # unitless, cxb scale factor (slow-type fibre)
            'r_sl':  0.126, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

            # Bioenergetics parameters 
            'c_c_tot': 29.5, # mM, Kushmerick et al. 1992 
            'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992 
            'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992 
            'V_max_oxphos': 2 * 2.0255, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 0.742, # unitless, # original
            'r_rec': 0.5 * 0.165e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 1, # Scaling factor for metabolic rates at rest    

            # Other model parameters
            # Konno et al., 2025 model parameters 
            'r1': 0.6177,
            'r2': 0.2342,

        }, 
        'EDL': {

            # Excitation-activation parameters
            'Tau_1': 0.011,  # requested
            'Tau_2': 0.011,  # BH 2003, fibre bundle data
            "K": 0.45,
            "n": 2.89, # Hill coefficient for activation model

            # Mechanical parameters
            'F_0': 0, # N,
            'l_0': 8.9e-3, # m,
            'mass': 3.9e-3, # g,
            'max_iso_stress': 3.01e5, # N/m^2, B1996
            'dedt_ce_max': 11, # s^-1, Barclay 2010
            'kappa': 0.29,

            # Initial energetics model
            'r_cxb':   0.763, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cat': 0.0199, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'cxb_scale':  0.257, # unitless, cxb scale factor
            'r_sl':  0.105, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

            # Bioenergetics parameters
            'c_c_tot': 29.5, # mM, Kushmerick et al. 1992
            'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992
            'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992
            'V_max_oxphos': 2 * 2.0255, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 0.742, # unitless, # original
            'r_rec': 0.5 * 0.165e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 1, # Scaling factor for metabolic rates at rest    

            # Other model parameters
            # Konno et al., 2025 model parameters
            'r1': 2.7919,
            'r2': 0.697,
        },

        # Other bioenergetics parameters
        # Assume constant across all species and muscle fibre-types
        'V_ck_f': 100,# 100, # mM/s, Kushmerick 1998
        'K_b': 1.11, #mM, MacFarland 1994
        'K_ia': 0.135, # mM, MacFarland 1994
        'K_eq': 1.77e2, # Assuming a pH of 7, Lawson 1979
        'K_iq': 3.5, # mM, MacFarland 1994
        'K_ib': 3.9, # mM, MacFarland 1994
        'K_p': 3.8, # mM, MacFarland 1994
        'Gatp': 60e3, # J/mol, Free energy of ATP (Barclay 2019)

        # Mechanical model SEE stiffness (unused)
        'k_see': 0, 
}


# Compute additional parameters
for muscle in ('SOL', 'EDL'):
    # Maximum isometric force
    params[muscle]['F_0'] = (
        params[muscle]['mass'] / params['rho0'] /
        params[muscle]['l_0'] *
        params[muscle]['max_iso_stress']
    )


# Function to compute  the stimulation
def f_stim(t, stim_freq = 100, contr_freq = 1, N_cycles = 1, duty_factor = 0.5): 
    '''
    Function to compute the stimulation times for the simulation
    '''
    # Compute the time per cycle
    cycle_length =  1 / contr_freq # Get the time with respect to the cycle
    t_stim_end = cycle_length * duty_factor # Get the time 
    t_cycle = t % cycle_length # Get the time with respect to the cycle

    # Toggle whether in stimulation or not (does not define frequency of stim here)
    stim = ((t_cycle <= t_stim_end) ) * (t < cycle_length * N_cycles)

    # Compute the stimulation times 
    stim_times = np.zeros_like(t, dtype=int)

    # Compute the period 
    period = 1.0 / stim_freq

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
            
    return stim, stim_times


# Function to run the model 
def run_model(t_vec, params):
    '''
    This function takes as input the time vector and parameter and runs the model. 
    
    Returns time varying heat rates 
    '''
    # Define the muscle we are using
    muscle = params['muscle']

    # Compute the stimulation times for this frequency
    stim, stim_times_vec = f_stim(
        t_vec,
        stim_freq = params['stim_freq'], 
        contr_freq = params['contr_freq'], 
        N_cycles = params['N_cycles'],
        duty_factor = params['duty_factor'] 
    )

    # Ca dynamics
    act_model = ActivationModel(params[muscle], t_vec)
    idx_stims = np.nonzero(stim_times_vec)[0]
    stim_vec, ca_vec, catn_vec = act_model.runExcAct(idx_stims)

    # Mechanics
    mech_model = MechModel(params[muscle]['l_0'], params[muscle]['dedt_ce_max'], params[muscle]['kappa'], params['k_see'])
    e_ce = np.zeros_like(t_vec)
    dedt_ce = np.zeros_like(t_vec)
    force_direct = mech_model.computeForce(catn_vec, e_ce + 1, dedt_ce)

    # Initial energetics
    energy_model = EnergeticsModel()
    q_a, q_m, q_sl, w = energy_model.solveInitialEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force_direct, mech_model)
    E_tot = q_a + q_m + q_sl + w
    E_initial_converted = E_tot * params[muscle]['F_0'] * params[muscle]['l_0'] / params[muscle]['mass']

    # Bioenergetics solve
    bioenergetic_model = Bioenergetics(params)
    t_span = (t_vec[0], t_vec[-1])
    c_atp_0 = params[muscle]['c_atp_0']
    sol = bioenergetic_model.solveBioenergetics(t_span, c_atp_0, t_vec, E_initial_converted)

    # Recovery energetics
    scale = params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0']
    q_r = bioenergetic_model.computeRecoveryEnergetics(sol.t, sol.y[0,]) * scale # Units of F0l0/s
    
    return q_a, q_m, q_sl, w, q_r # F0l0/s

# Define the time vector (s) 
dt = 0.0001
t_vec = np.linspace(params['t_start'], params['t_end'], int((params['t_end'] - params['t_start']) / dt)) 

if params['sim_type'] == 'varycontrfreq':
    #_____________________________________________________________________
    # Option 1: Influence of contraction frequency on muscle energetics
    #           Plots the time course and recovery time constants
    print('Varying contraction frequencies...')

    # Define the list of contraction frequencies 
    contr_freq_list = params['contr_freq_list']

    # Define data storage arrays 
    results = []
    component_cum_time_series = []

    # This section loops over simulation frequencies 
    for idx, contr_freq in enumerate(contr_freq_list):
        color = palette[idx % len(palette)]
        print(f'Running for contraction frequency {contr_freq}')

        # Set the contraction frequency in the parameters file 
        params['contr_freq'] = contr_freq

        # Run the model 
        q_a, q_m, q_sl, w, q_r  = run_model(t_vec, params) 
        total_energy_rate = q_a + q_m + q_sl + w + q_r
        initial_energy_rate = total_energy_rate - q_r

        # Compute the total energies 
        q_a_cum = cumulative_trapezoid(q_a, t_vec, initial=0)
        q_m_cum = cumulative_trapezoid(q_m, t_vec, initial=0)
        q_sl_cum = cumulative_trapezoid(q_sl, t_vec, initial=0)
        w_cum = cumulative_trapezoid(w, t_vec, initial=0)
        q_r_cum = cumulative_trapezoid(q_r, t_vec, initial=0)
        total_energy_cum = q_a_cum + q_m_cum + q_sl_cum + w_cum + q_r_cum
        component_cum_time_series.append((contr_freq, q_a_cum.copy(), q_m_cum.copy(), q_sl_cum.copy(), w_cum.copy(), q_r_cum.copy(), total_energy_cum.copy()))

        # Fit recovery decay with a 3 s post-stimulus buffer.
        t_contr_end = 1 / contr_freq * params['N_cycles'] # Get the end time of the contractions
        mask = t_vec >= (t_contr_end + 3.0)
        if np.count_nonzero(mask) < 20:
            print(f'Skipping tau fit for {contr_freq} Hz (insufficient decay points).')
            results.append((contr_freq, np.nan, np.nan, np.nan))
            continue

        t_decay = t_vec[mask]
        y_decay = total_energy_rate[mask]
        t_rel = t_decay - t_decay[0]

        def exp_decay(t, y_inf, A, tau):
            return y_inf + A * np.exp(-t / tau)

        tail_n = min(500, len(y_decay))
        y_inf_guess = float(np.mean(y_decay[-tail_n:]))
        A_guess = float(y_decay[0] - y_inf_guess)
        tau_guess = 20.0

        p0 = (y_inf_guess, A_guess, tau_guess)
        bounds = ([-np.inf, -np.inf, 1e-9], [np.inf, np.inf, np.inf])
        try:
            popt, _ = curve_fit(exp_decay, t_rel, y_decay, p0=p0, bounds=bounds, maxfev=20000)
            tau_fit = float(popt[2])
        except RuntimeError:
            tau_fit = np.nan
            print(f'Tau fit failed for {contr_freq} Hz')


        E_init_end = float(cumulative_trapezoid(total_energy_cum - q_r, t_vec, initial=0)[-1])
        E_rec_end = float(cumulative_trapezoid(q_r, t_vec, initial=0)[-1])
        rec_over_init = E_rec_end / E_init_end if np.abs(E_init_end) > 1e-12 else np.nan
        results.append((contr_freq, tau_fit, rec_over_init))

elif params['sim_type'] == 'varystimfreq':
    #_____________________________________________________________________
    # Influence of stimulation frequency on muscle energetics
    #           Plots the time course and recovery time constants
    print('Varying stimulation frequencies...')

    # Define the list of contraction frequencies 
    stim_freq_list = params['stim_freq_list']

    # Define data storage arrays 
    results = []
    component_cum_time_series = []

    # This section loops over simulation frequencies 
    for idx, stim_freq in enumerate(stim_freq_list):
        color = palette[idx % len(palette)]
        print(f'Running for contraction frequency {stim_freq}')

        # Set the contraction frequency in the parameters file 
        params['stim_freq'] = stim_freq

        # Run the model 
        q_a, q_m, q_sl, w, q_r  = run_model(t_vec, params) 
        total_energy_rate = q_a + q_m + q_sl + w + q_r
        initial_energy_rate = total_energy_rate - q_r


        # Compute the total energies 
        q_a_cum = cumulative_trapezoid(q_a, t_vec, initial=0)
        q_m_cum = cumulative_trapezoid(q_m, t_vec, initial=0)
        q_sl_cum = cumulative_trapezoid(q_sl, t_vec, initial=0)
        w_cum = cumulative_trapezoid(w, t_vec, initial=0)
        q_r_cum = cumulative_trapezoid(q_r, t_vec, initial=0)
        total_energy_cum = q_a_cum + q_m_cum + q_sl_cum + w_cum + q_r_cum
        component_cum_time_series.append((stim_freq, q_a_cum.copy(), q_m_cum.copy(), q_sl_cum.copy(), w_cum.copy(), q_r_cum.copy(), total_energy_cum.copy()))

        # Fit recovery decay with a 3 s post-stimulus buffer.
        t_contr_end = 1 / stim_freq * params['N_cycles'] # Get the end time of the contractions
        mask = t_vec >= (t_contr_end + 3.0)
        if np.count_nonzero(mask) < 20:
            print(f'Skipping tau fit for {stim_freq} Hz (insufficient decay points).')
            results.append((stim_freq, np.nan, np.nan, np.nan))
            continue

        t_decay = t_vec[mask]
        y_decay = total_energy_rate[mask]
        t_rel = t_decay - t_decay[0]

        def exp_decay(t, y_inf, A, tau):
            return y_inf + A * np.exp(-t / tau)

        tail_n = min(500, len(y_decay))
        y_inf_guess = float(np.mean(y_decay[-tail_n:]))
        A_guess = float(y_decay[0] - y_inf_guess)
        tau_guess = 20.0

        p0 = (y_inf_guess, A_guess, tau_guess)
        bounds = ([-np.inf, -np.inf, 1e-9], [np.inf, np.inf, np.inf])
        try:
            popt, _ = curve_fit(exp_decay, t_rel, y_decay, p0=p0, bounds=bounds, maxfev=20000)
            tau_fit = float(popt[2])
        except RuntimeError:
            tau_fit = np.nan
            print(f'Tau fit failed for {stim_freq} Hz')


        E_init_end = float(cumulative_trapezoid(total_energy_cum - q_r, t_vec, initial=0)[-1])
        E_rec_end = float(cumulative_trapezoid(q_r, t_vec, initial=0)[-1])
        rec_over_init = E_rec_end / E_init_end if np.abs(E_init_end) > 1e-12 else np.nan
        results.append((stim_freq, tau_fit, rec_over_init))

elif params['sim_type'] == 'single':
    # Protocol to simulate only one contraction using the parameter values
    print('Simulating a single contraction \n Frequency values are for the contraction frequency')
    results = []
    component_cum_time_series = []

    # Run the model 
    q_a, q_m, q_sl, w, q_r  = run_model(t_vec, params) 
    total_energy_rate = q_a + q_m + q_sl + w + q_r
    initial_energy_rate = total_energy_rate - q_r


    # Compute the total energies 
    q_a_cum = cumulative_trapezoid(q_a, t_vec, initial=0)
    q_m_cum = cumulative_trapezoid(q_m, t_vec, initial=0)
    q_sl_cum = cumulative_trapezoid(q_sl, t_vec, initial=0)
    w_cum = cumulative_trapezoid(w, t_vec, initial=0)
    q_r_cum = cumulative_trapezoid(q_r, t_vec, initial=0)
    total_energy_cum = q_a_cum + q_m_cum + q_sl_cum + w_cum + q_r_cum
    component_cum_time_series.append((params['contr_freq'], q_a_cum.copy(), q_m_cum.copy(), q_sl_cum.copy(), w_cum.copy(), q_r_cum.copy(), total_energy_cum.copy()))

    # Fit recovery decay with a 3 s post-stimulus buffer.
    t_contr_end = 1 / params['contr_freq'] * params['N_cycles'] # Get the end time of the contractions
    mask = t_vec >= (t_contr_end + 3.0)

    t_decay = t_vec[mask]
    y_decay = total_energy_rate[mask]
    t_rel = t_decay - t_decay[0]

    def exp_decay(t, y_inf, A, tau):
        return y_inf + A * np.exp(-t / tau)

    tail_n = min(500, len(y_decay))
    y_inf_guess = float(np.mean(y_decay[-tail_n:]))
    A_guess = float(y_decay[0] - y_inf_guess)
    tau_guess = 20.0

    p0 = (y_inf_guess, A_guess, tau_guess)
    bounds = ([-np.inf, -np.inf, 1e-9], [np.inf, np.inf, np.inf])
    try:
        popt, _ = curve_fit(exp_decay, t_rel, y_decay, p0=p0, bounds=bounds, maxfev=20000)
        tau_fit = float(popt[2])
    except RuntimeError:
        tau_fit = np.nan
        print(f'Tau fit failed for {params["contr_freq"]} Hz')


    E_init_end = float(cumulative_trapezoid(total_energy_cum - q_r, t_vec, initial=0)[-1])
    E_rec_end = float(cumulative_trapezoid(q_r, t_vec, initial=0)[-1])
    rec_over_init = E_rec_end / E_init_end if np.abs(E_init_end) > 1e-12 else np.nan
    results.append((params['contr_freq'], tau_fit, rec_over_init))
else: 
    print('Invalid sim_type!')



#_____________________________________________________________________
# Plots 

# Plot the time course of energy consumption 
fig_ts_all, ax_ts_all = plt.subplots(layout='constrained', figsize = (7,5))
for freq, q_a_cum, q_m_cum, q_sl_cum, w_cum, q_r_cum, total_energy_cum in component_cum_time_series:
    color = palette[int(np.log2(freq)) % len(palette)]
    label = f'{freq} Hz'
    ax_ts_all.plot(t_vec, total_energy_cum, label=f'{label}', color=color, linestyle='-')
    ax_ts_all.plot(t_vec, q_r_cum, label=None, color=color, linestyle='--')
    ax_ts_all.plot(t_vec, total_energy_cum - q_r_cum, label=None, color=color, linestyle=':')

ax_ts_all.set_xlabel('Time ($s$)')
ax_ts_all.set_ylabel('Cumulative energy ($F_0 l_0$)')
ax_ts_all.legend(ncol=2, loc = 'upper right')

# Plot the energetic components 
freqs = []
q_a_end = []
q_m_end = []
q_sl_end = []
w_end = []
q_r_end = []
total_end = []
for (freq, q_a_cum, q_m_cum, q_sl_cum, w_cum, q_r_cum, total_cum) in component_cum_time_series:
    freqs.append(freq)
    q_a_end.append(q_a_cum[-1])
    q_m_end.append(q_m_cum[-1])
    q_sl_end.append(q_sl_cum[-1])
    w_end.append(w_cum[-1])
    q_r_end.append(q_r_cum[-1])
    total_end.append(total_cum[-1])

fig_comp_end, ax_comp_end = plt.subplots(layout='constrained')
ax_comp_end.plot(freqs, q_a_end, '-o', label='q_a')
ax_comp_end.plot(freqs, q_m_end, '-o', label='q_m')
ax_comp_end.plot(freqs, q_sl_end, '-o', label='q_sl')
ax_comp_end.plot(freqs, w_end, '-o', label='w')
ax_comp_end.plot(freqs, q_r_end, '-o', label='q_r')
ax_comp_end.set_xscale('log')
ax_comp_end.set_xlabel('Frequency ($Hz$)')
ax_comp_end.set_ylabel('Total energy spent ($F_0 l_0$)')
ax_comp_end.legend()

#_____________________________________________________________________
# Print results to a table
header_1 = '{:>7} | {:>5} | {:>13}'.format('freq_Hz', 'tau_s', 'E_rec/E_init')
separator_1 = '{:-<7}-+-{:-<5}-+-{:-<13}'.format('', '', '')
print(header_1)
print(separator_1)
for stim_freq, tau_fit, rec_over_init in results:
    print(f'{stim_freq:7.2f} | {tau_fit:5.2f} | {rec_over_init:13.4f}')


print('')
print('Integrated component outputs (mJ/g)')
header_2 = '{:>7} | {:>5} | {:>5} | {:>6} | {:>5} | {:>5}'.format('freq_Hz', 'q_a', 'q_m', 'q_sl', 'w', 'q_r')
separator_2 = '{:-<7}-+-{:-<5}-+-{:-<5}-+-{:-<6}-+-{:-<5}-+-{:-<5}'.format('', '', '', '', '', '')
print(header_2)
print(separator_2)
for row in component_cum_time_series:
    print(f'{row[0]:7.2f} | {row[1][-1]:5.3f} | {row[2][-1]:5.3f} | {row[3][-1]:6.3f} | {row[4][-1]:5.3f} | {row[5][-1]:5.3f}')


plt.show()