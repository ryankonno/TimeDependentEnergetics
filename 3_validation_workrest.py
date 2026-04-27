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
    't_start': 0, # s
    't_end': 100, # s (1 s work + 100 s recovery)
    'target_energy_first_second_j': 1.0, # J, set to desired X Joules
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


            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # Best simulations values gamma = 3 and gamma = 1
            # #__________
            # # Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, BUGFIXED!
            # # 'V_max_oxphos': 0.94548, # mM/s
            # 'V_max_oxphos': 2 * 1.49397, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            # 'K_adp': 0.058, # mM,
            # 'nh': 0.3156, # unitless, # original
            # # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            # 'r_rec': 0.045887e6, # J / mol, Obtained from efficiency calculation 
            # 'gamma': 3, # Scaling factor for metabolic rates at rest   
            #__________
            # Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, BUGFIXED!
            # 'V_max_oxphos': 0.94548, # mM/s
            'V_max_oxphos': 1.9322, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 0.61325, # unitless, # original
            # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            'r_rec': 0.16730e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 1, # Scaling factor for metabolic rates at rest   
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%          

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
            # Q10
            'r_cxb':  0.3786, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cat': 0.0662, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_sl': 0.239, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

        }, 
        'EDL': { 
            'c_c_tot': 29.5, # mM, Kushmerick et al. 1992 
            'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992 
            'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992 

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #__________
            # SOL VALUES Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, BUGFIXED!
            # 'V_max_oxphos': 0.94548, # mM/s
            'V_max_oxphos': 2 * 1.49397, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 0.3156, # unitless, # original
            # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            'r_rec': 0.045887e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 3, # Scaling factor for metabolic rates at rest         

            # #__________
            # # SOL VALUES Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, BUGFIXED!
            # # 'V_max_oxphos': 0.94548, # mM/s
            # 'V_max_oxphos': 1.9322, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            # 'K_adp': 0.058, # mM,
            # 'nh': 0.61325, # unitless, # original
            # # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            # 'r_rec': 0.16730e6, # J / mol, Obtained from efficiency calculation 
            # 'gamma': 1, # Scaling factor for metabolic rates at rest   

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





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
            'dedt_ce_max': 11, 
            'kappa': 0.25,

            # Energetics model 
            # Q10
            'r_cxb':  2.439, # F0l0/s, Maximum heat rate of isometric contraction (fast-type fibre)
            'r_cat': 0.1497, # F0l0/s, Maximum heat rate of isometric contraction (fast-type fibre)
            'r_sl': 1.0146, # W/F_0/l_0, Maximum shortening heat rate (fast-type fibre)

        },

        # Assume constant across all species and muscle fibre-types
        'V_ck_f': 100,# 100, # mM/s, Kushmerick 1998
        'K_b': 1.11, #mM, MacFarland 1994
        'K_ia': 0.135, # mM, MacFarland 1994
        'K_eq': 1.77e2, # unitless, Assuming a pH of 7, Lawson 1979
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
Run the model 
'''

# Plot to verify conditions 
t_vec = np.linspace(params['t_start'], params['t_end'], int(10000 * params['t_end']))

from Models.BioenergeticsSimple import Bioenergetics

muscle = params['muscle']
mass_g = params[muscle]['mass']
work_duration_s = 1.0
target_energy_values_j = np.linspace(70, 350, 10) * 1e-3  # mW/g approx values from BW2004

results = []

fig_energy, ax_energy = plt.subplots(layout='constrained')
fig_tau, ax_tau = plt.subplots(layout='constrained')
fig_metab, (ax_atp, ax_pcr) = plt.subplots(2, 1, sharex=True, layout='constrained')

for i, target_energy_j in enumerate(target_energy_values_j):
    color = palette[i % len(palette)]
    print(f'Running target energy {target_energy_j}')
    x_label = f'X = {target_energy_j * 1e3:.2f} mJ'

    # Convert target first-second energy [J] into input ATP-use drive [W/g].
    E_initial_converted = np.zeros_like(t_vec)
    input_power_w_per_g = target_energy_j 
    E_initial_converted[t_vec < work_duration_s] = input_power_w_per_g

    # Solve bioenergetics
    bioenergetic_model = Bioenergetics(params)
    t_span = (t_vec[0], t_vec[-1])
    c_atp_0 = params[muscle]['c_atp_0']
    sol = bioenergetic_model.solveBioenergetics(t_span, c_atp_0, t_vec, E_initial_converted)

    # ATP and PCr concentrations from ODE state [mM].
    c_atp = sol.y[0,]
    c_pcr = sol.y[1,]

    # Recovery power from model [W/g], and total power [mW/g] for fitting.
    q_r_w_per_g = bioenergetic_model.computeRecoveryEnergetics(sol.t, sol.y[0,])
    total_energy_rate_mw_per_g = (E_initial_converted + q_r_w_per_g) * 1e3

    # Convert rates to absolute muscle power [W] for integration in Joules.
    e_init_w = E_initial_converted * mass_g
    q_r_w = q_r_w_per_g * mass_g

    e_init_j = cumtrapz(e_init_w, t_vec, initial=0)
    q_r_j = cumtrapz(q_r_w, t_vec, initial=0)
    total_j = e_init_j + q_r_j

    ax_energy.plot(t_vec, total_j, color=color, label=x_label)

    # Fit recovery decay from t >= 1 s.
    mask = t_vec >= work_duration_s + 2
    t_decay = t_vec[mask]
    y_decay = total_energy_rate_mw_per_g[mask]
    t_rel = t_decay - t_decay[0]

    def exp_decay(t, y_inf, A, tau):
        return y_inf + A * np.exp(-t / tau)

    tail_n = min(500, len(y_decay))
    y_inf_guess = float(np.mean(y_decay[-tail_n:]))
    A_guess = float(y_decay[0] - y_inf_guess)
    tau_guess = 20.0

    p0 = (y_inf_guess, A_guess, tau_guess)
    bounds = ([-np.inf, -np.inf, 1e-9], [np.inf, np.inf, np.inf])
    popt, _ = curve_fit(exp_decay, t_rel, y_decay, p0=p0, bounds=bounds, maxfev=20000)
    tau_fit = float(popt[2])

    ax_tau.plot(t_rel, y_decay, color=color, alpha=0.25)
    ax_tau.plot(t_rel, exp_decay(t_rel, *popt), '--', color=color, label=f'{x_label}, tau = {tau_fit:.2f} s')

    ax_atp.plot(sol.t, c_atp, color=color, label=x_label)
    ax_pcr.plot(sol.t, c_pcr, color=color, label=x_label)

    e_init_end_j = float(e_init_j[-1])
    q_r_end_j = float(q_r_j[-1])
    if np.abs(e_init_end_j) > 1e-12:
        recovery_to_init_ratio = q_r_end_j / e_init_end_j
    else:
        recovery_to_init_ratio = np.nan

    results.append((target_energy_j, tau_fit, e_init_end_j, q_r_end_j, recovery_to_init_ratio))

ax_energy.set_xlabel('Time (s)')
ax_energy.set_ylabel('Integrated energy (J)')
ax_energy.legend()

ax_tau.set_xlabel('Time since recovery start (s)')
ax_tau.set_ylabel('Energy rate ($mW g^{-1}$)')
ax_tau.legend()

ax_atp.set_ylabel('ATP (mM)')
# ax_atp.legend()
ax_pcr.set_xlabel('Time (s)')
ax_pcr.set_ylabel('PCr (mM)')
# ax_pcr.legend()

for target_energy_j, tau_fit, e_init_end_j, q_r_end_j, recovery_to_init_ratio in results:
    print(
        f'X = {target_energy_j:.4f} J | tau = {tau_fit:.3f} s | '
        f'init_energy = {e_init_end_j:.4f} J | recovery_energy = {q_r_end_j:.4f} J | '
        f'recovery/initial = {recovery_to_init_ratio:.4f}'
    )