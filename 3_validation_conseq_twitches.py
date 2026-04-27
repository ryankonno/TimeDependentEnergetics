'''
Code the validate the recovery model using data from Mast and Elzinga 1987

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

from Models.MUActivationModel import ActivationModel
from Models.MechanicsModelSimple import MechModel 
from Models.MUEnergeticsModelSimple_SplitVars import EnergeticsModel

# Parameters
params = {
    # Time parameters for setting up the protocol 
    't_start': 0, # s, 10 s pre-stimulation buffer
    't_end': 2, # s
    'cycle_length': 0.3, # s, Defines the length of the cycle (used to set frequency of the contractions)
    'N_cycles': 10, # unitless, Number of cycles to simulate (rest period after N_cycles contractions)

    # General muscle parameters
    'rho0':  1e6,    # g/m^3, Density of muscle
    'max_iso_stress': 2.5e5, # N/m^2, Maximum isometric stress of the muscle

    'muscle': 'SOL', # Specify muscle parameters to be used in simulation

        # Mouse data 
        'SOL': {

            'c_c_tot': 29.5, # mM, Kushmerick et al. 1992 
            'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992 
            'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992 

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Best simulations values gamma = 3 and gamma = 1
            #__________
            # Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, BUGFIXED!
            # 'V_max_oxphos': 0.94548, # mM/s
            'V_max_oxphos': 1.49397, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 0.3156, # unitless, # original
            # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            'r_rec': 0.045887e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 3, # Scaling factor for metabolic rates at rest   
            # #__________
            # # Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, BUGFIXED!
            # # 'V_max_oxphos': 0.94548, # mM/s
            # 'V_max_oxphos': 1.9322, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            # 'K_adp': 0.058, # mM,
            # 'nh': 0.61325, # unitless, # original
            # # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            # 'r_rec': 0.5 * 0.16730e6, # J / mol, Obtained from efficiency calculation 
            # 'gamma': 1, # Scaling factor for metabolic rates at rest   
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%          

            # Values from Barclay and Weber 2004
            'F_0': 0, # N, 
            'l_0': 11e-3, # m, 
            'mass': 4.1e-3, # g, 
            'max_iso_stress': 2.88e5, # N/m^2, B2012

            'Tau_1': 0.011,  # requested
            'Tau_2': 0.032,  # BH 2003, fibre bundle data
            "K": 0.1025,
            "n": 2.89, # Hill coefficient for activation model

            
            # # "Tau_1": 0.0422, # Assume constant value from MCL (2023)
            # "Tau_1": 0.0007, # BH2012 @ 30deg
            # # "Tau_1":  0.011 * 0.5, # BH2012 @ 20deg (assuming half release with tem (B2012))
            # # "Tau_2": 0.125, # Scaling based on MCL (2023)
            # # "Tau_2": 0.057, #  BH 2012        
            # # "Tau_2": 0.065, # B2012 20deg
            # "Tau_2": 0.013, # B2012 30deg
            # # # "Tau_2": 0.5, #  BH 2012
            # # 'Tau_1': 0.096,# BH 2012 mouse 20deg
            # # 'Tau_2': 0.130,# BH 2012 mouse 20deg
            
            # # "K": 0.1025, # Mayfield
            # "K": 0.035,
            # "n": 2.89, # Hill coefficient for act mdoel



            
            # Mechanical parameters 
            'dedt_ce_max': 5, 
            'kappa': 0.18,

            # Initial energetics model 
            # Q10
            # 
            # 'r_cxb':  0.3786, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_cat': 0.0662, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_sl': 0.239, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)
            # # Values using tau_1 = 0.038
            # 'r_cxb':  0.37656, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_cat': 0.0614, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_sl': 0.23774, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)
            # # Values accounting for temperature at 20degrees 
            # 'r_cxb':  0.092766, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_cat': 0.024266, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_sl': 0.05857, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)
            # # Based on twitch data ( 0.004ms twitch)
            # 'r_cxb': 0.4283, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_cat': 0.05439, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_sl': 0.2342, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)
            # # Based on twitch data ( 0.001ms twitch)
            # 'r_cxb': 179.31489, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_cat': 0.4647, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_sl': 0.2342, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)
            # Based on twitch data ( 0.001ms twitch), new K
            'r_cxb': 2.783, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cat': 0.4647, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_sl': 0.2342, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)
            

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

for muscle in ('SOL',):
    # Maximum isometric force (Assuming fixed max_iso_stress for now)    
    params[muscle]['F_0'] = params[muscle]['mass'] / params['rho0'] / params[muscle]['l_0'] * params[muscle]['max_iso_stress']
    print(f'{muscle}: Maximum isometric stress: {params[muscle]["F_0"]}')

'''
Setup the simulation 
'''
def f_stim_length(t, params, n_twitches=2, freq_hz=0.5): 
    # Function to compute the length changes in the muscle 
    # returns both simulation times and lengths 

    t_stim_start = 0.0
    freq = freq_hz  # Hz
    period = 1.0 / freq

    # Change in length (mm)
    dl = np.zeros_like(t)

    # Build an explicit twitch train for the requested twitch count/frequency.
    t_fire_vec = t_stim_start + np.arange(n_twitches) * period
    t_stim_end = t_fire_vec[-1]

    # Toggle whether in stimulation or not.
    stim = ((t >= t_stim_start) & (t <= t_stim_end)).astype(int)

    # stim_times: vector (same shape as t) with 1 where a stimulus occurs, 0 otherwise
    stim_times = np.zeros_like(t, dtype=int)
    if t_fire_vec.size > 0:
        t_arr = np.asarray(t)
        for st in t_fire_vec:
            idx = int(np.argmin(np.abs(t_arr - st)))
            stim_times[idx] = 1
            
    return stim, stim_times, dl


'''
Run the model 
'''

# Plot to verify conditions 
t_vec = np.linspace(params['t_start'], params['t_end'], int(10000 * (params['t_end'] - params['t_start']))) 

n_twitches = 2
freq_list = (1.0, 2.0, 4, 8, 16, 32, 64, 128, 256, 512)

results = []
peak_qr_vs_freq = []
activation_ratio_rows = []
ca_release_ratio_rows = []

fig_ca, ax_ca = plt.subplots(layout='constrained')
fig_act_ratio, ax_act_ratio = plt.subplots(layout='constrained')
fig_ca_release_ratio, ax_ca_release_ratio = plt.subplots(layout='constrained')

# Figure for time-varying energy components
fig_energy_components, axs_energy_comp = plt.subplots(
    5, 1, layout='constrained', figsize=(10, 12),
    sharex=True
)

component_colors = (
    '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e'
)

for idx, stim_freq_hz in enumerate(freq_list):
    color = palette[idx % len(palette)]
    print(f'Running {n_twitches} twitches at {stim_freq_hz} Hz')

    # Compute the stimulation times for this frequency.
    stim_protocol_vec, stim_times_vec, dl_vec = f_stim_length(
        t_vec,
        params,
        n_twitches=n_twitches,
        freq_hz=stim_freq_hz,
    )

    # Ca dynamics
    act_model = ActivationModel(params[params['muscle']], t_vec, True)
    idx_stims = np.nonzero(stim_times_vec)[0]
    stim_vec, ca_vec, catn_vec = act_model.runExcAct(idx_stims, w_0=0.004)
    ax_ca.plot(t_vec, ca_vec, color=color, label=f'{stim_freq_hz} Hz')

    # Mechanics
    muscle = params['muscle']
    mech_model = MechModel(params[muscle]['l_0'], params[muscle]['dedt_ce_max'], params[muscle]['kappa'], params['k_see'])
    e_ce = dl_vec / params[muscle]['l_0'] + 0.1
    dedt_ce = np.diff(e_ce, prepend=0) / np.diff(t_vec, prepend=1)
    force_direct = mech_model.computeForce(catn_vec, e_ce + 1, dedt_ce)

    # Initial energetics
    energy_model = EnergeticsModel()
    q_a, q_m, q_sl, w = energy_model.actEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force_direct, mech_model)
    E_tot = q_a + q_m + q_sl + w
    E_initial_converted = E_tot * params[muscle]['F_0'] * params[muscle]['l_0'] / params[muscle]['mass']

    # One-twitch reference activation heat at the same frequency.
    _, stim_times_single, _ = f_stim_length(t_vec, params, n_twitches=1, freq_hz=stim_freq_hz)
    idx_stims_single = np.nonzero(stim_times_single)[0]
    _, ca_single, catn_single = act_model.runExcAct(idx_stims_single, w_0=0.004)
    q_a_single, _ = energy_model.actEnergetics(t_vec, ca_single, catn_single, params[muscle])

    # Relative Ca release using the same subtraction approach.
    dt = np.diff(t_vec, prepend=t_vec[0])
    dt[dt == 0] = 1e-12
    ca_release_rate_two = np.maximum(np.diff(ca_vec, prepend=ca_vec[0]) / dt, 0.0)
    ca_release_rate_one = np.maximum(np.diff(ca_single, prepend=ca_single[0]) / dt, 0.0)
    ca_release_two_twitch = float(np.trapz(ca_release_rate_two, t_vec))
    ca_release_one_twitch = float(np.trapz(ca_release_rate_one, t_vec))
    ca_release_second = ca_release_two_twitch - ca_release_one_twitch
    pct_ca_second_vs_first = 100.0 * ca_release_second / ca_release_one_twitch if np.abs(ca_release_one_twitch) > 1e-12 else np.nan
    ca_release_ratio_rows.append((stim_freq_hz, ca_release_one_twitch, ca_release_two_twitch, ca_release_second, pct_ca_second_vs_first))

    # Activation heat by subtraction:
    # second twitch heat = (total activation heat of two twitches) - (activation heat of one twitch).
    q_a_two_twitch = float(np.trapz(q_a, t_vec))
    q_a_one_twitch = float(np.trapz(q_a_single, t_vec))
    q_a_second = q_a_two_twitch - q_a_one_twitch
    pct_second_vs_first = 100.0 * q_a_second / q_a_one_twitch if np.abs(q_a_one_twitch) > 1e-12 else np.nan
    activation_ratio_rows.append((stim_freq_hz, q_a_one_twitch, q_a_two_twitch, q_a_second, pct_second_vs_first))

    # Bioenergetics solve
    from Models.BioenergeticsSimple import Bioenergetics
    bioenergetic_model = Bioenergetics(params)
    t_span = (t_vec[0], t_vec[-1])
    c_atp_0 = params[muscle]['c_atp_0']
    sol = bioenergetic_model.solveBioenergetics(t_span, c_atp_0, t_vec, E_initial_converted)

    # Recovery energetics
    scale = params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0']
    q_r = bioenergetic_model.computeRecoveryEnergetics(
        sol.t, sol.y[0,]
    ) * scale
    energy_unit_scaler = (
        params[muscle]['F_0'] * params[muscle]['l_0'] /
        params[muscle]['mass'] * 1e3
    )

    # Plot energy components for first frequency
    if idx == 0:
        # Compute cumulative energy for each component
        q_a_cum = cumtrapz(
            q_a, t_vec, initial=0
        ) * energy_unit_scaler
        q_m_cum = cumtrapz(
            q_m, t_vec, initial=0
        ) * energy_unit_scaler
        q_sl_cum = cumtrapz(
            q_sl, t_vec, initial=0
        ) * energy_unit_scaler
        w_cum = cumtrapz(
            w, t_vec, initial=0
        ) * energy_unit_scaler
        q_r_cum = cumtrapz(
            q_r, t_vec, initial=0
        ) * energy_unit_scaler
        
        # Plot on separate subplots
        axs_energy_comp[0].plot(
            t_vec, q_a_cum,
            label='$q_a$ (activation)',
            color=component_colors[0], linewidth=2
        )
        axs_energy_comp[1].plot(
            t_vec, q_m_cum,
            label='$q_m$ (maintenance)',
            color=component_colors[1], linewidth=2
        )
        axs_energy_comp[2].plot(
            t_vec, q_sl_cum,
            label='$q_{sl}$ (shortening)',
            color=component_colors[2], linewidth=2
        )
        axs_energy_comp[3].plot(
            t_vec, w_cum,
            label='$w$ (mechanical work)',
            color=component_colors[3], linewidth=2
        )
        axs_energy_comp[4].plot(
            t_vec, q_r_cum,
            label='$q_r$ (recovery)',
            color=component_colors[4], linewidth=2
        )
        
        # Add labels and formatting
        for i, (ax, label) in enumerate(zip(
            axs_energy_comp,
            ['Activation Heat', 'Maintenance Heat',
             'Shortening Heat', 'Mechanical Work',
             'Recovery Heat']
        )):
            ax.set_ylabel('Energy (mJ/g)')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        axs_energy_comp[-1].set_xlabel('Time (s)')
        fig_energy_components.suptitle(
            f'{muscle} muscle: '\
            f'{n_twitches} twitches at {stim_freq_hz} Hz',
            fontsize=12, fontweight='bold'
        )
    total_energy_rate = (E_tot + q_r) * energy_unit_scaler

    # Fit recovery decay with a 3 s post-stimulus buffer.
    t_stim_end = (n_twitches - 1) / stim_freq_hz
    mask = t_vec >= (t_stim_end + 3.0)
    if np.count_nonzero(mask) < 20:
        print(f'Skipping tau fit for {stim_freq_hz} Hz (insufficient decay points).')
        results.append((stim_freq_hz, np.nan, np.nan, np.nan))
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
        print(f'Tau fit failed for {stim_freq_hz} Hz')

    peak_qr = float(np.max(q_r[mask] * energy_unit_scaler))
    peak_qr_vs_freq.append((stim_freq_hz, peak_qr))

    E_tot_end = float(cumtrapz(E_tot, t_vec, initial=0)[-1])
    E_rec_end = float(cumtrapz(q_r, t_vec, initial=0)[-1])
    rec_over_init = E_rec_end / E_tot_end if np.abs(E_tot_end) > 1e-12 else np.nan
    results.append((stim_freq_hz, tau_fit, peak_qr, rec_over_init))

ax_ca.set_xlabel('Time (s)')
ax_ca.set_ylabel('Ca concentration (normalised)')
ax_ca.set_title('Ca Concentration')
ax_ca.legend()

activation_ratio_arr = np.array(activation_ratio_rows, dtype=float)
between_twitch_time = 1.0 / activation_ratio_arr[:, 0]
ax_act_ratio.plot(between_twitch_time, activation_ratio_arr[:, 4], '-o', color=palette[0], label='Second / First twitch (subtraction method)')
ax_act_ratio.set_xlabel('Between-twitch time (s)')
ax_act_ratio.set_ylabel('Activation heat (%)')
ax_act_ratio.set_title('Second-twitch Activation Heat Percentage')
ax_act_ratio.set_xscale('log')
ax_act_ratio.grid(True, alpha=0.3)
ax_act_ratio.legend()

ca_release_ratio_arr = np.array(ca_release_ratio_rows, dtype=float)
between_twitch_time_ca = 1.0 / ca_release_ratio_arr[:, 0]
ax_ca_release_ratio.plot(between_twitch_time_ca, ca_release_ratio_arr[:, 4], '-o', color=palette[2], label='Second / First Ca release (subtraction method)')
ax_ca_release_ratio.set_xlabel('Between-twitch time (s)')
ax_ca_release_ratio.set_ylabel('Relative Ca release (%)')
ax_ca_release_ratio.set_title('Second-twitch Relative Ca Release')
ax_ca_release_ratio.set_xscale('log')
ax_ca_release_ratio.grid(True, alpha=0.3)
ax_ca_release_ratio.legend()

print('freq_Hz | tau_s | peak_qr_mWg | E_rec/E_init')
print('--------+-------+-------------+-----------')
for stim_freq_hz, tau_fit, peak_qr, rec_over_init in results:
    print(f'{stim_freq_hz:7.2f} | {tau_fit:5.2f} | {peak_qr:11.4f} | {rec_over_init:9.4f}')

print('')
print('Activation heat ratios')
print('freq_Hz | qA_one | qA_two | qA_second = two-one | second/first_%')
print('--------+--------+--------+---------------------+---------------')
for row in activation_ratio_rows:
    print(f'{row[0]:7.2f} | {row[1]:6.4f} | {row[2]:6.4f} | {row[3]:19.4f} | {row[4]:13.2f}')

print('')
print('Ca release ratios')
print('freq_Hz | Ca_one | Ca_two | Ca_second = two-one | second/first_%')
print('--------+--------+--------+---------------------+---------------')
for row in ca_release_ratio_rows:
    print(f'{row[0]:7.2f} | {row[1]:6.4f} | {row[2]:6.4f} | {row[3]:19.4f} | {row[4]:13.2f}')

plt.show()