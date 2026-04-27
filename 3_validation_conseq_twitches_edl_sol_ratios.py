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

# Parameters
params = {
    # Time parameters for setting up the protocol
    't_start': -10,  # s, 10 s pre-stimulation buffer
    't_end': 50,  # s
    'cycle_length': 0.3,  # s, Defines the length of the cycle (used to set frequency of the contractions)
    'N_cycles': 10,  # unitless, Number of cycles to simulate (rest period after N_cycles contractions)

    # General muscle parameters
    'rho0': 1e6,  # g/m^3, Density of muscle
    'max_iso_stress': 2.5e5,  # N/m^2, Maximum isometric stress of the muscle

    'muscle': 'SOL',  # Placeholder, overwritten in muscle loop

    # Mouse data
    'SOL': {
        'c_c_tot': 29.5,  # mM, Kushmerick et al. 1992
        'c_atp_0': 5.3,  # mM, Kushmerick et al. 1992
        'c_pcr_0': 21.1,  # mM, Kushmerick et al. 1992

        'V_max_oxphos': 1.49397,  # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
        'K_adp': 0.058,  # mM
        'nh': 0.3156,  # unitless, original
        'r_rec': 0.045887e6,  # J / mol, Obtained from efficiency calculation
        'gamma': 3,  # Scaling factor for metabolic rates at rest

        'F_0': 0,  # N
        'l_0': 11e-3,  # m
        'mass': 4.1e-3,  # g
        'max_iso_stress': 2.88e5,  # N/m^2, B2012

        'Tau_1': 0.001,  # requested
        'Tau_2': 0.055,  # B2012 30deg
        'K': 0.035,
        'n': 1.99,

        'dedt_ce_max': 5,
        'kappa': 0.18,

        'r_cxb': 2.783,  # F0l0/s
        'r_cat': 0.4647,  # F0l0/s
        'r_sl': 0.2342,  # W/F_0/l_0
    },
    'EDL': {
        'c_c_tot': 29.5,  # mM, Kushmerick et al. 1992
        'c_atp_0': 5.3,  # mM, Kushmerick et al. 1992
        'c_pcr_0': 21.1,  # mM, Kushmerick et al. 1992

        'V_max_oxphos': 1.49397,  # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
        'K_adp': 0.058,  # mM
        'nh': 0.3156,  # unitless, original
        'r_rec': 0.045887e6,  # J / mol, Obtained from efficiency calculation
        'gamma': 3,  # Scaling factor for metabolic rates at rest

        'F_0': 0,  # N
        'l_0': 8.9e-3,  # m
        'mass': 3.9e-3,  # g
        'max_iso_stress': 36.0e3 * 5,  # N/m^2, B2012

        'Tau_1': 0.0007,  # requested
        'Tau_2': 0.097,  # BH 2003, fibre bundle data
        'K': 0.1,
        'n': 2.89,

        'dedt_ce_max': 10,
        'kappa': 0.29,

        'r_cxb': 2.7946,  # F0l0/s
        'r_cat': 0.288,  # F0l0/s
        'r_sl': 0.697,  # W/F_0/l_0
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
        stim_vec, ca_vec, catn_vec = act_model.runExcAct(idx_stims, w_0=0.001)

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
        _, ca_single, catn_single = act_model.runExcAct(idx_stims_single, w_0=0.001)
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
freq_list = (1.0, 2.0, 4, 8, 16, 32, 64, 128, 256, 512)

sol_act_rows, sol_ca_rows = analyze_muscle('SOL', t_vec, freq_list, n_twitches=2)
edl_act_rows, edl_ca_rows = analyze_muscle('EDL', t_vec, freq_list, n_twitches=2)

fig_act_ratio, ax_act_ratio = plt.subplots(layout='constrained')
fig_ca_release_ratio, ax_ca_release_ratio = plt.subplots(layout='constrained')

sol_between_twitch = 1.0 / sol_act_rows[:, 0]
edl_between_twitch = 1.0 / edl_act_rows[:, 0]

ax_act_ratio.plot(sol_between_twitch, sol_act_rows[:, 4], '-o', color='#1f77b4', label='SOL (tau_1 = 0.001)')
ax_act_ratio.plot(edl_between_twitch, edl_act_rows[:, 4], '-o', color='#d62728', label='EDL (tau_1 = 0.0007)')
ax_act_ratio.set_xlabel('Between-twitch time (s)')
ax_act_ratio.set_ylabel('Activation heat (%)')
ax_act_ratio.set_title('Second-twitch Activation Heat Percentage')
ax_act_ratio.set_xscale('log')
ax_act_ratio.grid(True, alpha=0.3)
ax_act_ratio.legend()

ax_ca_release_ratio.plot(sol_between_twitch, sol_ca_rows[:, 4], '-o', color='#1f77b4', label='SOL (tau_1 = 0.001)')
ax_ca_release_ratio.plot(edl_between_twitch, edl_ca_rows[:, 4], '-o', color='#d62728', label='EDL (tau_1 = 0.0007)')
ax_ca_release_ratio.set_xlabel('Between-twitch time (s)')
ax_ca_release_ratio.set_ylabel('Relative Ca release (%)')
ax_ca_release_ratio.set_title('Second-twitch Relative Ca Release')
ax_ca_release_ratio.set_xscale('log')
ax_ca_release_ratio.grid(True, alpha=0.3)
ax_ca_release_ratio.legend()

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

plt.show()
