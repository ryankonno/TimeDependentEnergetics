
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
import lib.plot_style

# global plotting defaults are set in lib.plot_style
import matplotlib.cm as cmap
palette = ("#32cd9c", "#f67410", "#2b21b8", "#C21599", "#83d921", "#1ab6e9")
import itertools
import sys 
sys.path.append('./')

from Models.ActivationModel import ActivationModel
from Models.MechanicsModel import MechModel 
from Models.InitialEnergeticsModel import EnergeticsModel
# Parameters
params = {
    # Time parameters for setting up the protocol 
    't_start': 0, # s
    't_end': 2, # s
    'cycle_length': 0.3, # s, Defines the length of the cycle (used to set frequency of the contractions)
    'N_cycles': 10, # unitless, Number of cycles to simulate (rest period after N_cycles contractions)

    # General muscle parameters
    'rho0':  1e6,    # g/m^3, Density of muscle
    'max_iso_stress': 2.5e5, # N/m^2, Maximum isometric stress of the muscle

    'muscle': 'EDL', # Specify muscle parameters to be used in simulation

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
    # Maximum isometric force (Assuming fixed max_iso_stress for now)    
    params[muscle]['F_0'] = params[muscle]['mass'] / params['rho0'] / params[muscle]['l_0'] * params[muscle]['max_iso_stress']
    print(f'{muscle}: Maximum isometric stress: {params[muscle]["F_0"]}')

'''
Setup the simulation 
'''
def f_stim_length(t, params, n_twitches=150, freq_hz=150): 
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
t_vec = np.linspace(params['t_start'], params['t_end'], int(100000 * (params['t_end'] - params['t_start']))) 

n_twitches = (1.0, 2.0, 4, 8, 16, 32, 64, 128, 256, 512)
freq_list = (1.0, 2.0, 4, 8, 16, 32, 64, 128, 256, 512)

results = []
peak_qr_vs_freq = []
peak_phi_atp_vs_freq = []
heat_time_series = []
component_heat_rows = []
component_cum_time_series = []

fig_ca, ax_ca = plt.subplots(layout='constrained')
fig_heat_time, ax_heat_time = plt.subplots(layout='constrained')
fig_component_heat, ax_component_heat = plt.subplots(layout='constrained')
fig_components_time, axs_components_time = plt.subplots(5, 1, layout='constrained', sharex=True, figsize = (5,20))

for idx, stim_freq_hz in enumerate(freq_list):
    color = palette[idx % len(palette)]
    print(f'Running {n_twitches[idx]} twitches at {stim_freq_hz} Hz')

    # Compute the stimulation times for this frequency.
    stim_protocol_vec, stim_times_vec, dl_vec = f_stim_length(
        t_vec,
        params,
        n_twitches=n_twitches[idx],
        freq_hz=stim_freq_hz,
    )

    # Ca dynamics
    act_model = ActivationModel(params[params['muscle']], t_vec)
    idx_stims = np.nonzero(stim_times_vec)[0]
    stim_vec, ca_vec, catn_vec = act_model.runExcAct(idx_stims)
    ax_ca.plot(t_vec, ca_vec, color=color, label=f'{stim_freq_hz} Hz')

    # Mechanics
    muscle = params['muscle']
    mech_model = MechModel(params[muscle]['l_0'], params[muscle]['dedt_ce_max'], params[muscle]['kappa'], params['k_see'])
    e_ce = dl_vec / params[muscle]['l_0'] + 0.1
    dedt_ce = np.diff(e_ce, prepend=0) / np.diff(t_vec, prepend=1)
    force_direct = mech_model.computeForce(catn_vec, e_ce + 1, dedt_ce)

    # Initial energetics
    energy_model = EnergeticsModel()
    q_a, q_m, q_sl, w = energy_model.solveInitialEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force_direct, mech_model)
    E_tot = q_a + q_m + q_sl + w
    E_initial_converted = E_tot * params[muscle]['F_0'] * params[muscle]['l_0'] / params[muscle]['mass']

    # Bioenergetics solve
    from Models.BioenergeticsModel import Bioenergetics
    bioenergetic_model = Bioenergetics(params)
    t_span = (t_vec[0], t_vec[-1])
    c_atp_0 = params[muscle]['c_atp_0']
    sol = bioenergetic_model.solveBioenergetics(t_span, c_atp_0, t_vec, E_initial_converted)

    # Maximum ATP consumption rate (umol/g/s) at this stimulation frequency.
    phi_atp_vec = bioenergetic_model.phi_atp(sol.t, sol.y[0,])
    peak_phi_atp_vs_freq.append((stim_freq_hz, float(np.max(phi_atp_vec))))

    # Recovery energetics
    scale = params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0']
    q_r = bioenergetic_model.computeRecoveryEnergetics(sol.t, sol.y[0,]) * scale
    energy_unit_scaler = params[muscle]['F_0'] * params[muscle]['l_0'] / params[muscle]['mass'] * 1e3

    total_energy_rate = (E_tot + q_r) * energy_unit_scaler
    total_energy_cum = cumtrapz(total_energy_rate, t_vec, initial=0)
    heat_time_series.append((stim_freq_hz, total_energy_cum.copy()))

    # Store cumulative component heat time courses (mJ/g) for multi-panel plotting.
    q_a_rate = q_a * energy_unit_scaler
    q_m_rate = q_m * energy_unit_scaler
    q_sl_rate = q_sl * energy_unit_scaler
    w_rate = w * energy_unit_scaler
    q_r_rate = q_r * energy_unit_scaler
    q_a_cum = cumtrapz(q_a_rate, t_vec, initial=0)
    q_m_cum = cumtrapz(q_m_rate, t_vec, initial=0)
    q_sl_cum = cumtrapz(q_sl_rate, t_vec, initial=0)
    w_cum = cumtrapz(w_rate, t_vec, initial=0)
    q_r_cum = cumtrapz(q_r_rate, t_vec, initial=0)
    component_cum_time_series.append((stim_freq_hz, q_a_cum.copy(), q_m_cum.copy(), q_sl_cum.copy(), w_cum.copy(), q_r_cum.copy()))

    # Integrated component outputs over the simulation window (mJ/g).
    q_a_end_mJg = float(cumtrapz(q_a, t_vec, initial=0)[-1] * energy_unit_scaler)
    q_m_end_mJg = float(cumtrapz(q_m, t_vec, initial=0)[-1] * energy_unit_scaler)
    q_sl_end_mJg = float(cumtrapz(q_sl, t_vec, initial=0)[-1] * energy_unit_scaler)
    w_end_mJg = float(cumtrapz(w, t_vec, initial=0)[-1] * energy_unit_scaler)
    q_r_end_mJg = float(cumtrapz(q_r, t_vec, initial=0)[-1] * energy_unit_scaler)
    component_heat_rows.append((stim_freq_hz, q_a_end_mJg, q_m_end_mJg, q_sl_end_mJg, w_end_mJg, q_r_end_mJg))

    # Fit recovery decay with a 3 s post-stimulus buffer.
    t_stim_end = (n_twitches[idx] - 1) / stim_freq_hz
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

for idx, (stim_freq_hz, total_energy_cum) in enumerate(heat_time_series):
    color = palette[idx % len(palette)]
    ax_heat_time.plot(t_vec, total_energy_cum, color=color, label=f'{stim_freq_hz} Hz')

ax_heat_time.set_xlabel('Time (s)')
ax_heat_time.set_ylabel('Cumulative heat ($mJ g^{-1}$)')
ax_heat_time.set_title('Cumulative Total Heat by Frequency')
ax_heat_time.grid(True, alpha=0.3)
ax_heat_time.legend()

component_names = ('q_a', 'q_m', 'q_sl', 'w', 'q_r')
for idx, (stim_freq_hz, q_a_cum, q_m_cum, q_sl_cum, w_cum, q_r_cum) in enumerate(component_cum_time_series):
    color = palette[idx % len(palette)]
    component_series = (q_a_cum, q_m_cum, q_sl_cum, w_cum, q_r_cum)
    for comp_idx, series in enumerate(component_series):
        axs_components_time[comp_idx].plot(t_vec, series, color=color, label=f'{stim_freq_hz} Hz')

for comp_idx, comp_name in enumerate(component_names):
    axs_components_time[comp_idx].set_ylabel(f'{comp_name}\n($mJ g^{{-1}}$)')
    axs_components_time[comp_idx].grid(True, alpha=0.3)

axs_components_time[0].set_title('Cumulative Heat by Component and Frequency')
axs_components_time[-1].set_xlabel('Time (s)')
axs_components_time[0].legend(ncol=2, fontsize=9)

component_heat_arr = np.array(component_heat_rows, dtype=float)
ax_component_heat.plot(component_heat_arr[:, 0], component_heat_arr[:, 1], '-o', color=palette[0], label='q_a')
ax_component_heat.plot(component_heat_arr[:, 0], component_heat_arr[:, 2], '-o', color=palette[1], label='q_m')
ax_component_heat.plot(component_heat_arr[:, 0], component_heat_arr[:, 3], '-o', color=palette[2], label='q_sl')
ax_component_heat.plot(component_heat_arr[:, 0], component_heat_arr[:, 4], '-o', color=palette[3], label='w')
ax_component_heat.plot(component_heat_arr[:, 0], component_heat_arr[:, 5], '-o', color=palette[4], label='q_r')
ax_component_heat.set_xlabel('Stimulation frequency (Hz)')
ax_component_heat.set_ylabel('Energy ($mJ g^{-1}$)')
ax_component_heat.set_title('Energy by Component')
ax_component_heat.set_xscale('log')
ax_component_heat.grid(True, alpha=0.3)
ax_component_heat.legend()

print('freq_Hz | tau_s | peak_qr_mWg | E_rec/E_init')
print('--------+-------+-------------+-----------')
for stim_freq_hz, tau_fit, peak_qr, rec_over_init in results:
    print(f'{stim_freq_hz:7.2f} | {tau_fit:5.2f} | {peak_qr:11.4f} | {rec_over_init:9.4f}')

print('')
print('Cumulative heat traces plotted for each frequency level.')

print('')
print('Integrated component outputs (mJ/g)')
print('freq_Hz | q_a | q_m | q_sl | w | q_r')
print('--------+-----+-----+------+---+-----')
for row in component_heat_rows:
    print(f'{row[0]:7.2f} | {row[1]:5.3f} | {row[2]:5.3f} | {row[3]:6.3f} | {row[4]:3.3f} | {row[5]:5.3f}')

print('')
print('Maximum ATP consumption rate phi_atp (umol/g/s)')
print('freq_Hz | max_phi_atp')
print('--------+------------')
for stim_freq_hz, max_phi_atp in peak_phi_atp_vs_freq:
    print(f'{stim_freq_hz:7.2f} | {max_phi_atp:10.4f}')

plt.show()