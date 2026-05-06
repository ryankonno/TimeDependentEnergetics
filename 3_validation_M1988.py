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
    't_start': -10, # s, 10 s pre-stimulation buffer
    't_end': 100, # s
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
            'V_max_oxphos': 1.9322, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 0.61325, # unitless, # original
            # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            'r_rec': 0.5 * 0.16730e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 1, # Scaling factor for metabolic rates at rest    
            #__________
            # # Corrected values
            # 'V_max_oxphos': 2 * 1.9322, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            # 'K_adp': 0.058, # mM,
            # 'nh': 0.61325, # unitless, # original
            # # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            # 'r_rec': 0.0433e6, # J / mol, Obtained from efficiency calculation 
            # 'gamma': 1, # Scaling factor for metabolic rates at rest    


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
            # __________
            # # Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, BUGFIXED!
            # 'V_max_oxphos': 0.94548, # mM/s
            'V_max_oxphos': 2 * 1.9322, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 0.61325, # unitless, # original
            # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            'r_rec': 0.5 * 0.16730e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 1, # Scaling factor for metabolic rates at rest    
            # #__________
            # # Corrected values
            # 'V_max_oxphos': 2 * 1.9322, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            # 'K_adp': 0.058, # mM,
            # 'nh': 0.61325, # unitless, # original
            # # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            # 'r_rec': 0.0433e6, # J / mol, Obtained from efficiency calculation 
            # 'gamma': 1, # Scaling factor for metabolic rates at rest    

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
            'dedt_ce_max': 12, 
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
def f_stim_length(t, params): 
    # Function to compute the length changes in the muscle 
    # returns both simulation times and lengths 

    t_stim_start = 0.0
    n_twitches = 10
    freq = 0.2  # Hz
    period = 1.0 / freq

    # Change in length (mm)
    dl = np.zeros_like(t)

    # Build an explicit twitch train to guarantee exactly 120 stimuli at 0.5 Hz.
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

# Compute the stimulation times
stim_protocol_vec, stim_times_vec,  dl_vec = f_stim_length(t_vec, params)
single_run_idx = 0
stim_freq_hz = 0.5

# Containers were previously filled inside a loop over conditions.
component_energy_abs = []
peak_qr_vs_freq = []
efficiency_rows = []

# Ca dynamics
act_model = ActivationModel(params[params['muscle']], t_vec, True)
idx_stims = np.nonzero(stim_times_vec)[0]
stim_vec, ca_vec, catn_vec = act_model.runExcAct(idx_stims, w_0 = 0.001)
# Plot the results 
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, ca_vec, label = 'Free Ca') 
ax.plot(t_vec, catn_vec, label = 'CaTn')
# ax.plot(stim_vec)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Normalised concentration')
plt.show()


# Mechanics 
muscle = params['muscle']
mech_model = MechModel(params[muscle]['l_0'], params[muscle]['dedt_ce_max'], params[muscle]['kappa'],params['k_see'])
# Compute the strain and strain rates in the muscle 
e_ce = dl_vec / params[muscle]['l_0'] + 0.1 # Get the strain adjusted so length change is over plateau
# e_ce = dl_vec / params[muscle]['l_0'] + (params[params['muscle']]['max_dl']) / (2 * params[muscle]['l_0']) # Get the strain adjusted so length change is over plateau
dedt_ce = np.diff(e_ce, prepend = 0) / np.diff(t_vec, prepend = 1)

# Compute the force directly  
force_direct =  mech_model.computeForce(catn_vec, e_ce + 1, dedt_ce)

# Plot force trace.
fig_force, ax_force = plt.subplots(layout = 'constrained')
ax_force.plot(t_vec, force_direct, color = palette[single_run_idx], label = 'Force')
ax_force.set_xlabel('Time (s)')
ax_force.set_ylabel('Normalised force')
ax_force.legend()

# Plot one-cycle force trace using time normalised by cycle length.
cycle_mask = (t_vec >= 0) * (t_vec < params['cycle_length'])
t_cycle = t_vec[cycle_mask]
t_cycle_norm = t_cycle / params['cycle_length']

# Compute the initial energetics 
energy_model = EnergeticsModel()
q_a, q_m, q_sl, w = energy_model.actEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force_direct, mech_model)
E_tot = q_a + q_m + q_sl + w  # F0l0/s, Total energy 
# Plot the rates 
# fig, ax = plt.subplots(layout = 'constrained')
# ax.plot(t_vec, q_a, label = '$\dot q_a$') 
# ax.plot(t_vec, q_m, label = '$\dot q_m$') 
# ax.plot(t_vec, q_sl, label = '$\dot q_{sl}$') 
# ax.plot(t_vec, w, label = '$\dot w$')
# ax.legend()
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Energy rate ($W \, (F_0 l_0)^{-1}$)')
# # Plot the total energy over the cycle
# fig, ax = plt.subplots(layout = 'constrained')
# ax.plot(t_vec, cumtrapz(q_a, t_vec, initial = 0), label = '$ q_a$') 
# ax.plot(t_vec, cumtrapz(q_m, t_vec, initial = 0), label = '$ q_m$') 
# ax.plot(t_vec, cumtrapz(q_sl, t_vec, initial = 0), label = '$ q_{sl}$') 
# ax.plot(t_vec, cumtrapz(w, t_vec, initial = 0), label = '$ w$')
# ax.legend()
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Energy  ($J \, (F_0 l_0)^{-1}$)')

# Convert units to input for bioenergetics model 
E_initial_converted = E_tot * params[muscle]['F_0'] * params[muscle]['l_0'] / params[muscle]['mass'] # W/g

# Run bioenergetics
from Models.BioenergeticsSimple import Bioenergetics
bioenergetic_model = Bioenergetics(params) 
t_span = (t_vec[0], t_vec[-1]) 
c_atp_0 = params[muscle]['c_atp_0']
# Solve the model
sol = bioenergetic_model.solveBioenergetics(t_span, c_atp_0, t_vec, E_initial_converted)

# Plot ATP and PCR concentrations in separate panels.
fig_atp_pcr_conc, axs_atp_pcr_conc = plt.subplots(2, 1, layout='constrained', sharex=True)
axs_atp_pcr_conc[0].plot(sol.t, sol.y[0,], color=palette[0], label='ATP')
axs_atp_pcr_conc[0].set_ylabel('ATP (mM)')
axs_atp_pcr_conc[0].set_title('ATP concentration')
axs_atp_pcr_conc[0].legend(loc='upper right')
axs_atp_pcr_conc[1].plot(sol.t, sol.y[1,], color=palette[1], label='PCR')
axs_atp_pcr_conc[1].set_xlabel('Time (s)')
axs_atp_pcr_conc[1].set_ylabel('PCR (mM)')
axs_atp_pcr_conc[1].set_title('PCR concentration')
axs_atp_pcr_conc[1].legend(loc='upper right')

# Plot ATP and PCR derivatives over time.
d_atp_dt = bioenergetic_model.atp_rhs(sol.t, sol.y)
d_pcr_dt = bioenergetic_model.pcr_rhs(sol.t, sol.y)
fig_atp_pcr, ax_atp_pcr = plt.subplots(layout='constrained')
ax_atp_pcr.plot(sol.t, d_atp_dt, label='dATP/dt', color=palette[0])
ax_atp_pcr.plot(sol.t, d_pcr_dt, label='dPCR/dt', color=palette[1])
ax_atp_pcr.set_xlabel('Time (s)')
ax_atp_pcr.set_ylabel('Rate (mM/s)')
ax_atp_pcr.set_title('ATP and PCR derivatives')
ax_atp_pcr.legend(loc='upper right')

# Compute the energetic rates 
scale =  params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0'] 
q_r = bioenergetic_model.computeRecoveryEnergetics(sol.t, sol.y[0,]) * scale # F0l0/s = J/g/s * g/F0l0

# Compute the scaler to get correct units
energy_unit_scaler = params[muscle]['F_0'] * params[muscle]['l_0'] / params[muscle]['mass'] * 1e3 # convert from F0l0/s to mW/g 

# Plot the total energy over the cycle
fig_energy, ax_energy = plt.subplots(layout = 'constrained')
ax_energy.plot(t_vec, cumtrapz(E_tot, t_vec, initial = 0) * energy_unit_scaler, label = '$ e_{init}$', color = palette[single_run_idx], alpha = 0.25) 
ax_energy.plot(t_vec, cumtrapz(q_r, t_vec, initial = 0) * energy_unit_scaler, label = '$ q_r$', color = palette[single_run_idx], ls = ':', alpha = 0.5) 
ax_energy.plot(t_vec, cumtrapz(E_tot + q_r, t_vec, initial = 0) * energy_unit_scaler, label = '$ q_r + e_{init}$', color = palette[single_run_idx]) 
# ax_energy.legend()
ax_energy.set_xlabel('Time (s)')
ax_energy.set_ylabel('Energy  ($mJ g^{-1}$)')
# fig_energy.savefig('./Figures/B2004_SepVars_EnergyUse_' + params['muscle'] + '.jpg')
# fig_energy.savefig('./Figures/B2004_SepVars_EnergyUse_' + params['muscle'] + '.svg')

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

# Plot total heat rate over the full trial.
fig_heat_rate, ax_heat_rate = plt.subplots(layout = 'constrained')
ax_heat_rate.plot(t_vec, total_energy_rate, color = palette[single_run_idx], label = 'Total heat rate')
ax_heat_rate.plot(t_vec, E_tot * energy_unit_scaler, color = palette[single_run_idx], ls = ':', label = 'Intialheat rate')
ax_heat_rate.plot(t_vec, q_r * energy_unit_scaler, color = palette[single_run_idx], ls = '--', label = 'Recovery heat rate')
ax_heat_rate.set_xlabel('Time (s)')
ax_heat_rate.set_ylabel('Heat rate ($mW g^{-1}$)')
ax_heat_rate.set_ylim((0,0.75))
ax_heat_rate.legend()

mask = t_vec >= 50 + 3 # Add 3s buffer
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

fig_tau, ax_tau = plt.subplots(layout = 'constrained')
ax_tau.plot(t_rel, y_decay, color = palette[single_run_idx], alpha = 0.35, label = f'{stim_freq_hz} Hz decay')
ax_tau.plot(t_rel, exp_decay(t_rel, *popt), '--', color = palette[single_run_idx], label = f'{stim_freq_hz} Hz fit ($\\tau$ = {tau_fit:.2f} s)')
ax_tau.set_xlabel('Time since recovery start (s)')
ax_tau.set_ylabel('Energy rate ($mW g^{-1}$)')
ax_tau.legend()

# Compute the peak recovery rate 
peak_qr_vs_freq.append(np.max(q_r[mask] * energy_unit_scaler))

# Compute efficiencies from integrated energies over the full simulation window.
E_tot_end = cumtrapz(E_tot, t_vec, initial = 0)[-1]
E_rec_end = cumtrapz(q_r, t_vec, initial = 0)[-1]
W_end = cumtrapz(w, t_vec, initial = 0)[-1]
total_heat_end_mJg = cumtrapz(total_energy_rate, t_vec, initial = 0)[-1]

if np.isclose(E_rec_end, 0.0):
    init_to_recovery_heat_ratio = np.inf
else:
    init_to_recovery_heat_ratio = E_rec_end / E_tot_end

eta_init = W_end / E_tot_end
eta_total = W_end / (E_tot_end + E_rec_end)
# Ratio of recovery heat to initial energy: eta_init / eta_total - 1 = E_rec / E_tot
efficiency_ratio = (eta_init / eta_total - 1)
efficiency_rows.append((stim_freq_hz, eta_init, eta_total, efficiency_ratio))

print(f'Fitted time constant (tau) = {tau_fit:.3f} s')
print(f'Initial-to-recovery heat ratio (E_rec/E_init) = {init_to_recovery_heat_ratio:.4f}')
print(f'Total heat (integrated) = {total_heat_end_mJg:.4f} mJ/g')

plt.show()