'''
This script simulates the energetics of a 'work-rest' contraction. Despite the name this is an isometric contraction. 

The 'work' phase consists of a fixed initial energetic rate prescribed (E_rate). Following the initial period, the recovery of the energetic rates is computed using the bioenergetics model.

All parameters used by the model are defined in this file. The params dictionary contains the time settings used to configure the simulation together with muscle-specific energetics parameters for SOL and EDL.

The script produces three figures and prints a summary of the fitted recovery behaviour to the console:

    Figure 1: Cumulative total energy over the simulation

    Figure 2: Recovery energy rate after the work phase with the fitted exponential decay

    Figure 3: ATP and PCr concentration over time

    Table 1: The fitted time constant, initial energy, recovery energy, and recovery-to-initial energy ratio for each target energy value.

Ryan Konno
r.konno@uq.edu.au
The University of Queensland
'''


# Import 
import numpy as np 
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
import lib.plot_style
palette = ("#32cd9c", "#f67410", "#2b21b8", "#C21599", "#83d921", "#1ab6e9")
import sys 
sys.path.append('./')

from Models.BioenergeticsModel import Bioenergetics

# Define parameters
# note: could also import parameters from parameters_muscle.py
params = {
    # Time parameters for setting up the protocol 
    't_start': 0, # s
    't_end': 100, # s (1 s work + 100 s recovery)

        # General muscle parameters
    'rho0':  1e6, # g/m^3, Density of muscle

    'muscle': 'SOL',

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
            'cxb_scale':  0.567, # unitless, cxb scale factor
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

# Simulate contraction-rest simulations at various energetic rates 
target_energy_values_j = np.linspace(70, 350, 5) * 1e-3  # target energy rates during contraction

# Define the time
dt = 0.0001
t_vec = np.linspace(params['t_start'], params['t_end'], int((params['t_end'] - params['t_start']) / dt)) 

# Define the muscle 
muscle = params['muscle']
work_duration_s = 1.0


results = []

fig_energy, ax_energy = plt.subplots(layout='constrained')
fig_tau, ax_tau = plt.subplots(layout='constrained')
fig_metab, (ax_atp, ax_pcr) = plt.subplots(2, 1, sharex=True, layout='constrained')

for i, target_energy_j in enumerate(target_energy_values_j):
    color = palette[i % len(palette)]
    print(f'Running target energy rate {target_energy_j}')
    x_label = f'E_rate = {target_energy_j * 1e3:.0f} mJ/s'

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
    e_init_w = E_initial_converted * params[muscle]['mass']
    q_r_w = q_r_w_per_g * params[muscle]['mass']

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
ax_energy.set_ylabel('Energy (J)')
ax_energy.legend()

ax_tau.set_xlabel('Time since recovery start (s)')
ax_tau.set_ylabel('Energy rate ($mW g^{-1}$)')
ax_tau.legend()

ax_atp.set_ylabel('ATP (mM)')
# ax_atp.legend()
ax_pcr.set_xlabel('Time (s)')
ax_pcr.set_ylabel('PCr (mM)')
# ax_pcr.legend()

header = '{:>12} | {:>10} | {:>12} | {:>12} | {:>18}'.format(
    'E_rate (mJ/s)',
    'tau_s (s)',
    'E_init (J)',
    'E_rec (J)',
    'R (E_rec/E_init)',
)
separator = '{:-<12}-+-{:-<10}-+-{:-<12}-+-{:-<12}-+-{:-<18}'.format('', '', '', '', '')
print(header)
print(separator)
for target_energy_j, tau_fit, e_init_end_j, q_r_end_j, recovery_to_init_ratio in results:
    print(
        f'{target_energy_j * 1e3:12.2f} | {tau_fit:10.2f} | '
        f'{e_init_end_j:12.4f} | {q_r_end_j:12.4f} | {recovery_to_init_ratio:18.4f}'
    )