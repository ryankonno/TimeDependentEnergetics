'''
Consecutive twitch ratio analysis for SOL and EDL.

Uses the same ratio-based template as the consecutive twitch validation script,
but overlays both muscles on the same ratio plots.

SOL tau_1 = 0.001
EDL tau_1 = 0.0007

EDL parameters are taken from 3_validation_B2004_bothmuscles_newparams.py.
'''

# Import
import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

font = {'size': 14}
plt.rc('font', **font)
palette = ("#32cd9c", "#f67410", "#2b21b8", "#C21599", "#83d921", "#1ab6e9")

import sys
sys.path.append('./')

from Models.MUActivationModel import ActivationModel
from Models.MechanicsModelSimple import MechModel
from Models.MUEnergeticsModelSimple_SplitVars import EnergeticsModel

from lib.model_metrics import r2_score, mse_calc

# Parameters
params = {
    # Time parameters for setting up the protocol
    't_start': 0,  # s, 10 s pre-stimulation buffer
    't_end': 1,  # s
    'cycle_length': 0.3,  # s, Defines the length of the cycle (used to set frequency of the contractions)
    'N_cycles': 10,  # unitless, Number of cycles to simulate (rest period after N_cycles contractions)

    # General muscle parameters
    'rho0': 1e6,  # g/m^3, Density of muscle
    'max_iso_stress': 2.5e5,  # N/m^2, Maximum isometric stress of the muscle

    'muscle': 'SOL',  # Placeholder, overwritten in muscle loop

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
            'freq': 150, # Hz, Frequency of stimulation 
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
            # # Optimisation with a submax scaling factor, Cat NO scaling, B2010 data
            # 'r_cxb':  0.2473242, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_cat': 0.029479, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'cxb_scale':  0.5665, # unitless, cxb scale factor
            # 'r_sl':  0.26774, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)
            # Optimisation with a submax scaling factor, Cat NO scaling, B2010 data, r_s optimisation
            'r_cxb':   0.25843, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cat': 0.03540646954815866, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'cxb_scale':  0.51669189931185, # unitless, cxb scale factor
            'r_sl':  0.12584005987468994, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

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
            # 'freq': 160, # Hz, Frequency of stimulation, BW2004
            'freq': 200, # Hz, Frequency of stimulation, Adjusted for tetenanus
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
            # # Optimisation with a submax scaling factor , Cat NO scaling, B2010 FIT
            # 'r_cxb':  0.76267727, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'r_cat': 0.01992, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            # 'cxb_scale':  0.2565930, # unitless, cxb scale factor
            # 'r_sl':  0.697, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)
            # Optimisation with a submax scaling factor , Cat NO scaling, B2010 FIT, r_s optimisation
            'r_cxb':  0.761209, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cat': 0.0216, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'cxb_scale':  0.23276, # unitless, cxb scale factor
            'r_sl':  0.105056, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

            # Konno et al., 2025 model parameters 
            'r1': 2.7919,
            'r2': 0.697,


        },


    # Assume constant across all species and muscle fibre-types
    'V_ck_f': 100,
    'K_b': 1.11,
    'K_ia': 0.135,
    'K_eq': 1.77e2,
    'K_iq': 3.5,
    'K_ib': 3.9,
    'K_p': 3.8,
    'Gatp': 60e3,
    'k_see': 0,
}

# Compute necessary parameters from data
for muscle in ('SOL', 'EDL'):
    params[muscle]['F_0'] = params[muscle]['mass'] / params['rho0'] / params[muscle]['l_0'] * params[muscle]['max_iso_stress']
    print(f'{muscle}: Maximum isometric stress: {params[muscle]["F_0"]}')


def f_stim_length(t, params, n_twitches=2, freq_hz=0.5):
    """Build an explicit twitch train for the requested twitch count/frequency."""
    t_stim_start = 0.0
    freq = freq_hz
    period = 1.0 / freq

    dl = np.zeros_like(t)
    t_fire_vec = t_stim_start + np.arange(n_twitches) * period
    t_stim_end = t_fire_vec[-1]

    stim = ((t >= t_stim_start) & (t <= t_stim_end)).astype(int)

    stim_times = np.zeros_like(t, dtype=int)
    if t_fire_vec.size > 0:
        t_arr = np.asarray(t)
        for st in t_fire_vec:
            idx = int(np.argmin(np.abs(t_arr - st)))
            stim_times[idx] = 1

    return stim, stim_times, dl


def analyze_muscle(muscle_name, t_vec, freq_list, n_twitches=2):
    local_params = params.copy()
    local_params['muscle'] = muscle_name

    activation_ratio_rows = []
    ca_release_ratio_rows = []

    for stim_freq_hz in freq_list:
        print(f'Running {muscle_name} at {n_twitches} twitches, {stim_freq_hz} Hz')

        stim_protocol_vec, stim_times_vec, dl_vec = f_stim_length(
            t_vec,
            local_params,
            n_twitches=n_twitches,
            freq_hz=stim_freq_hz,
        )

        act_model = ActivationModel(local_params[local_params['muscle']], t_vec, True)
        idx_stims = np.nonzero(stim_times_vec)[0]
        stim_vec, ca_vec, catn_vec = act_model.runExcAct(idx_stims)

        muscle = local_params['muscle']
        mech_model = MechModel(
            local_params[muscle]['l_0'],
            local_params[muscle]['dedt_ce_max'],
            local_params[muscle]['kappa'],
            local_params['k_see'],
        )
        e_ce = dl_vec / local_params[muscle]['l_0'] + 0.1
        dedt_ce = np.diff(e_ce, prepend=0) / np.diff(t_vec, prepend=1)
        force_direct = mech_model.computeForce(catn_vec, e_ce + 1, dedt_ce)

        energy_model = EnergeticsModel()
        q_a, q_m, q_sl, w = energy_model.actEnergetics(
            t_vec,
            ca_vec,
            catn_vec,
            local_params[muscle],
            e_ce + 1,
            dedt_ce,
            force_direct,
            mech_model,
        )

        _, stim_times_single, _ = f_stim_length(t_vec, local_params, n_twitches=1, freq_hz=stim_freq_hz)
        idx_stims_single = np.nonzero(stim_times_single)[0]
        _, ca_single, catn_single = act_model.runExcAct(idx_stims_single)
        q_a_single, _ = energy_model.actEnergetics(t_vec, ca_single, catn_single, local_params[muscle])

        # Activation heat by subtraction.
        q_a_two_twitch = float(np.trapz(q_a, t_vec))
        q_a_one_twitch = float(np.trapz(q_a_single, t_vec))
        q_a_second = q_a_two_twitch - q_a_one_twitch
        pct_second_vs_first = 100.0 * q_a_second / q_a_one_twitch if np.abs(q_a_one_twitch) > 1e-12 else np.nan
        activation_ratio_rows.append((stim_freq_hz, q_a_one_twitch, q_a_two_twitch, q_a_second, pct_second_vs_first))

        # Ca release by subtraction.
        dt = np.diff(t_vec, prepend=t_vec[0])
        dt[dt == 0] = 1e-12
        ca_release_rate_two = np.maximum(np.diff(ca_vec, prepend=ca_vec[0]) / dt, 0.0)
        ca_release_rate_one = np.maximum(np.diff(ca_single, prepend=ca_single[0]) / dt, 0.0)
        ca_release_two_twitch = float(np.trapz(ca_release_rate_two, t_vec))
        ca_release_one_twitch = float(np.trapz(ca_release_rate_one, t_vec))
        ca_release_second = ca_release_two_twitch - ca_release_one_twitch
        pct_ca_second_vs_first = 100.0 * ca_release_second / ca_release_one_twitch if np.abs(ca_release_one_twitch) > 1e-12 else np.nan
        ca_release_ratio_rows.append((stim_freq_hz, ca_release_one_twitch, ca_release_two_twitch, ca_release_second, pct_ca_second_vs_first))

    return np.array(activation_ratio_rows, dtype=float), np.array(ca_release_ratio_rows, dtype=float)


# Run the model
# Plot to verify conditions
t_vec = np.linspace(params['t_start'], params['t_end'], int(10000 * (params['t_end'] - params['t_start'])))
freq_list = (2.0, 4, 8, 16, 32, 64, 128, 256, 512)

sol_act_rows, sol_ca_rows = analyze_muscle('SOL', t_vec, freq_list, n_twitches=2)
edl_act_rows, edl_ca_rows = analyze_muscle('EDL', t_vec, freq_list, n_twitches=2)

fig_act_ratio, ax_act_ratio = plt.subplots(figsize=(4,3))
fig_act_ratio.subplots_adjust(left=0.15)
fig_ca_release_ratio, ax_ca_release_ratio = plt.subplots(figsize=(4,3))
fig_ca_release_ratio.subplots_adjust(left=0.15)

sol_between_twitch = 1.0 / sol_act_rows[:, 0]
edl_between_twitch = 1.0 / edl_act_rows[:, 0]

# Import the experimental values 
import pandas as pd
data_exp_SOL = pd.read_csv('./Data/B2012_data_conseq_twitch_qA_SOL.csv')
data_exp_EDL = pd.read_csv('./Data/B2012_data_conseq_twitch_qA_EDL.csv')

ax_act_ratio.plot(np.array(data_exp_SOL['t_int']), np.array(data_exp_SOL['qA_rel']), ':o', color='#1f77b4', label='SOL Exp')
ax_act_ratio.plot(np.array(data_exp_EDL['t_int']), np.array(data_exp_EDL['qA_rel']), ':o', color='#d62728', label='EDL Exp')

ax_act_ratio.plot(sol_between_twitch, sol_act_rows[:, 4], '-o', color='#1f77b4', label='SOL Mod')   
ax_act_ratio.plot(edl_between_twitch, edl_act_rows[:, 4], '-o', color='#d62728', label='EDL Mod')
ax_act_ratio.set_xlabel('Between-twitch time (s)')
ax_act_ratio.set_ylabel('Activation heat (%)')
# ax_act_ratio.set_title('Second-twitch Activation Heat Percentage')
ax_act_ratio.set_xscale('log')
ax_act_ratio.grid(True, alpha=0.3)
ax_act_ratio.legend()
fig_act_ratio.savefig('Figures/conseq_twitches_activation_heat_ratio.jpg')
fig_act_ratio.savefig('Figures/conseq_twitches_activation_heat_ratio.svg')


# __________________________________
# Compute the metrics for r^2 and mse 
# Interpolate the model values to the experimental values 
sol_act_mod_interp = np.interp(np.array(data_exp_SOL['t_int']), np.flip(sol_between_twitch), np.flip(sol_act_rows[:, 4]))
edl_act_mod_interp = np.interp(np.array(data_exp_EDL['t_int']), np.flip(edl_between_twitch), np.flip(edl_act_rows[:, 4]))
r2_model_sol =  r2_score(sol_act_mod_interp, np.array(data_exp_SOL['qA_rel']))
mse_model_sol = mse_calc(sol_act_mod_interp, np.array(data_exp_SOL['qA_rel']))
r2_model_edl =  r2_score(edl_act_mod_interp, np.array(data_exp_EDL['qA_rel']))
mse_model_edl = mse_calc(edl_act_mod_interp, np.array(data_exp_EDL['qA_rel']))
print(f'SOL: r2 = {r2_model_sol}, mse = {mse_model_sol}')
print(f'EDL: r2 = {r2_model_edl}, mse = {mse_model_edl}')




ax_ca_release_ratio.plot(sol_between_twitch, sol_ca_rows[:, 4], '-o', color='#1f77b4', label='SOL (tau_1 = 0.001)')
ax_ca_release_ratio.plot(edl_between_twitch, edl_ca_rows[:, 4], '-o', color='#d62728', label='EDL (tau_1 = 0.0007)')
ax_ca_release_ratio.set_xlabel('Between-twitch time (s)')
ax_ca_release_ratio.set_ylabel('Relative Ca release (%)')
# ax_ca_release_ratio.set_title('Second-twitch Relative Ca Release')
ax_ca_release_ratio.set_xscale('log')
ax_ca_release_ratio.grid(True, alpha=0.3)
ax_ca_release_ratio.legend()
fig_ca_release_ratio.savefig('Figures/conseq_twitches_ca_release_ratio.jpg')
fig_ca_release_ratio.savefig('Figures/conseq_twitches_ca_release_ratio.svg')

# Plot absolute Ca released for consecutive twitches (first vs second)
fig_ca_consec, ax_ca_consec = plt.subplots(figsize=(4,3))
fig_ca_consec.subplots_adjust(left=0.15)
ax_ca_consec.plot(sol_between_twitch, sol_ca_rows[:, 1], '-o', color='#1f77b4', label='SOL first twitch')
ax_ca_consec.plot(sol_between_twitch, sol_ca_rows[:, 3], '--s', color='#1f77b4', label='SOL second twitch')
ax_ca_consec.plot(edl_between_twitch, edl_ca_rows[:, 1], '-o', color='#d62728', label='EDL first twitch')
ax_ca_consec.plot(edl_between_twitch, edl_ca_rows[:, 3], '--s', color='#d62728', label='EDL second twitch')
ax_ca_consec.set_xlabel('Between-twitch time (s)')
ax_ca_consec.set_ylabel('Ca released (integral)')
# ax_ca_consec.set_title('Ca Released: First vs Second Twitch')
ax_ca_consec.set_xscale('log')
ax_ca_consec.grid(True, alpha=0.3)
ax_ca_consec.legend()
fig_ca_consec.savefig('Figures/conseq_twitches_ca_released_consecutive.jpg')
fig_ca_consec.savefig('Figures/conseq_twitches_ca_released_consecutive.svg')

print('')
print('Activation heat ratios')
print('freq_Hz | SOL_second/first_% | EDL_second/first_%')
print('--------+--------------------+--------------------')
for i, freq in enumerate(freq_list):
    print(f'{freq:7.2f} | {sol_act_rows[i, 4]:18.2f} | {edl_act_rows[i, 4]:18.2f}')

print('')
print('Ca release ratios')
print('freq_Hz | SOL_second/first_% | EDL_second/first_%')
print('--------+--------------------+--------------------')
for i, freq in enumerate(freq_list):
    print(f'{freq:7.2f} | {sol_ca_rows[i, 4]:18.2f} | {edl_ca_rows[i, 4]:18.2f}')



# Time-varying Ca amount: plot Ca concentration over time for each stimulation frequency
colors = plt.cm.viridis(np.linspace(0, 1, len(freq_list)))

# SOL: time-varying Ca amount
fig_ca_time_sol, ax_ca_time_sol = plt.subplots(figsize=(4,3))
fig_ca_time_sol.subplots_adjust(left=0.15)
for ci, f in enumerate(freq_list):
    local_params = params.copy()
    local_params['muscle'] = 'SOL'
    stim_protocol_vec, stim_times_vec, dl_vec = f_stim_length(t_vec, local_params, n_twitches=2, freq_hz=f)
    act_model = ActivationModel(local_params[local_params['muscle']], t_vec, True)
    idx_stims = np.nonzero(stim_times_vec)[0]
    _, ca_vec, _ = act_model.runExcAct(idx_stims)
    ax_ca_time_sol.plot(t_vec, ca_vec, color=colors[ci], label=f'{int(f)} Hz')
ax_ca_time_sol.set_xlabel('Time (s)')
ax_ca_time_sol.set_ylabel('Ca amount')
# ax_ca_time_sol.set_title('SOL: Time-varying Ca')
ax_ca_time_sol.grid(True, alpha=0.3)
ax_ca_time_sol.legend()
fig_ca_time_sol.savefig('Figures/conseq_twitches_ca_time_SOL.jpg')
fig_ca_time_sol.savefig('Figures/conseq_twitches_ca_time_SOL.svg')

# EDL: time-varying Ca amount
fig_ca_time_edl, ax_ca_time_edl = plt.subplots(figsize=(4,3))
fig_ca_time_edl.subplots_adjust(left=0.15)
for ci, f in enumerate(freq_list):
    local_params = params.copy()
    local_params['muscle'] = 'EDL'
    stim_protocol_vec, stim_times_vec, dl_vec = f_stim_length(t_vec, local_params, n_twitches=2, freq_hz=f)
    act_model = ActivationModel(local_params[local_params['muscle']], t_vec, True)
    idx_stims = np.nonzero(stim_times_vec)[0]
    _, ca_vec, _ = act_model.runExcAct(idx_stims)
    ax_ca_time_edl.plot(t_vec, ca_vec, color=colors[ci], label=f'{int(f)} Hz')
ax_ca_time_edl.set_xlabel('Time (s)')
ax_ca_time_edl.set_ylabel('$c_{ca}$')
# ax_ca_time_edl.set_title('EDL: Time-varying Ca')
ax_ca_time_edl.grid(True, alpha=0.3)
# ax_ca_time_edl.legend()
fig_ca_time_edl.savefig('Figures/conseq_twitches_ca_time_EDL.jpg')
fig_ca_time_edl.savefig('Figures/conseq_twitches_ca_time_EDL.svg')

plt.show()