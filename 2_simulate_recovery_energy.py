'''
This is a code to simulate bioenergetics of muscle contraction from the conditions in Barclay et al., 1995

This version does not run the optimisation of the code, but just the simulation with given parameters (see 2_optimisation_recovery_energy.py for the code with the optimisation)

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

import sys 
sys.path.append('./')

# Change directory to save files
import os 
os.chdir('/mnt/c/Users/s4773677/uq_ws/TemporalEnergetics/Modelling/TimeDependentEnergy')

from Models.BioenergeticsModel import Bioenergetics

save_id = '_gamma3_optedlsol'

params = {
    # Time parameters for setting up the protocol 
    't_start': 0, # s
    't_end': 300, # s
    't_cycle_start': 10, # s
    't_cycle_end': 10 + 150, # s 
    'cycle_length': 1, # s, Defines the length of the cycle (used to set frequency of the contractions)
    'N_cycles': 10, # unitless, Number of cycles to simulate (rest period after N_cycles contractions)

    # General muscle parameters
    'rho0':  1e6,    # g/m^3, Density of muscle
    'max_iso_stress': 2.5e5, # N/m^2, Maximum isometric stress of the muscle

    'muscle': 'EDL', # Specify muscle parameters to be used in simulation

        # Mouse data 
        'SOL': {
            # Slow data 
            'c_c_tot': 25.9, # mM, Kushmerick et al. 1992 
            'c_atp_0': 3.3, # mM,  Kushmerick et al. 1992 
            'c_pcr_0': 11.4, # mM,  Kushmerick et al. 1992 
            # 'Pi_0': 6, # mM,  Kushmerick et al. 1992 
            # 'gamma': 1, # Scaling factor for metabolic rates at rest
            'atp_peak': 0.25,# 0.213, # mM/s 

            # # Values to match Ri/Rr = 1 and zero derivative 
            # 'V_max_oxphos': 3.47, # mM/s
            # 'K_adp': 0.0615, # mM,
            # 'nh': 0.873, # unitless, 
            # 'r_rec': 2.41e5, # F0 l0 / mol

            # #  recovery rate obtained from eff calc
            # 'V_max_oxphos': 1.88, # mM/s
            # 'K_adp': 0.058, # mM, 
            # 'nh': 2.57, # unitless,
            # 'r_rec': 43.3e3, # J / mol

            # # Values with recovery rate obtained from eff calc, and opt bioenergy 
            # 'V_max_oxphos': 5.24, # mM/s
            # 'K_adp': 0.315, # mM, 
            # 'nh': 1.00, # unitless,
            # 'r_rec': 43.3e3, # J / mol

            # # Values to match recovery rate during initial contractoin
            # 'V_max_oxphos': 1.88, # mM/s
            # 'K_adp': 0.058, # mM, 
            # 'nh': 2.57, # unitless,
            # 'r_rec': 0.1519e6, # J / mol

            # #__________
            # # Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, no r_rec opt
            # 'V_max_oxphos': 1.65548, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            # 'K_adp': 0.058, # mM,
            # 'nh': 0.2729, # unitless, # original
            # 'r_rec': 43.3e3, # J / mol, Obtained from efficiency calculation 
            # 'gamma': 3, # Scaling factor for metabolic rates at rest     

            #__________
            # Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, BUGFIXED!
            # 'V_max_oxphos': 0.94548, # mM/s
            'V_max_oxphos': 1.49397, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 0.3156, # unitless, # original
            # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            'r_rec': 0.045887e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 3, # Scaling factor for metabolic rates at rest        

            # Values from Barclay et al. 1995
            'F_0': 0, # N, 
            'l_0':  9.5e-3, # m, 
            'mass': 1.99e-3, # g, 
           
            
            
            # Heat data used for optimisation 
            # 'heat_exp_rec': np.array((5.339988e-03, 5.726164e-02, 7.235249e-02, 7.009909e-02)), # J/F0l0, Slow, recovery heat 
            # 'heat_exp_init': np.array((0.225, 0.176, 0.166, 0.164)),  # J/F0l0, Slow, initial heat
            'heat_exp_rec': np.array((4.478470e-03, 4.802343e-02, 6.067962e-02, 5.878977e-02)), # J/F0l0, Slow, recovery heat, ASSUME SAME 1/3 REC RATE DURING CONTRACTION
            'heat_exp_init': np.array((2.362291e-01, 2.190962e-01, 2.193518e-01, 2.148170e-01)),  # J/F0l0, Slow, initial heat, ASSUME SAME 1/3 REC RATE DURING CONTRACTION
        }, 
        'EDL': { 
            'c_c_tot': 29.5, # mM, Kushmerick et al. 1992 
            'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992 
            'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992 
            'Pi_0': 0, # mM,  Kushmerick et al. 1992 



            # # Values to match Ri/Rr = 1 and zero derivative 
            # 'V_max_oxphos': 3.47/2, # mM/s
            # 'K_adp': 0.0615, # mM,
            # 'nh': 0.873, # unitless, 
            # 'r_rec': 2.41e5, # F0 l0 / mol

            # #  Values with recovery rate obtained from eff calc
            # 'V_max_oxphos': 1.88/2, # mM/s
            # 'K_adp': 0.058, # mM, 
            # 'nh': 2.57, # unitless,
            # 'r_rec': 38.8e3, # J / mol

            # # Values with recovery rate obtained from eff calc, and opt bioenergy 
            # 'V_max_oxphos': 5.24, # mM/s
            # 'K_adp': 0.315, # mM, 
            # 'nh': 1.00, # unitless,
            # 'r_rec': 38.8e3, # J / mol

            # #  Values with opt recovery rate 
            # 'V_max_oxphos': 1.88/2, # mM/s
            # 'K_adp': 0.058, # mM, 
            # 'nh': 2.57, # unitless,
            # 'r_rec': 0.2165e6, # J / mol

            # #__________
            # # Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, no r_rec opt
            # # 'V_max_oxphos': 12.71738, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees 1.65548
            # 'V_max_oxphos': 0.5 * 1.65548 , # mM/s
            # 'K_adp': 0.058, # mM,
            # # 'nh': 0.56477, # unitless, # original
            # 'nh': 0.2729,
            # 'r_rec': 38.8e3, # J / mol, Obtained from efficiency calculation 
            # 'gamma': 3, # Scaling factor for metabolic rates at rest       
            #__________
            # Optimised values to B1995 (rrec, nh, vmax), gamma = 3, MEAN VALUE, scaled input data, BUGFIXED!
            # 'V_max_oxphos': 0.94548, # mM/s
            'V_max_oxphos': 0.5 * 1.49397, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 1.0593, # unitless, # original
            # 'r_rec': 0.06787e6, # J / mol, Obtained from efficiency calculation 
            'r_rec': 0.146329095e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 3, # Scaling factor for metabolic rates at rest          

            # For Phillips Simulation 
            'atp_peak': 0.25,# 0.213, # mM/s Peak atp rate calculated based on initial heat rate and enthalpy of ATP from Phillips et al. 1993
            # 'gamma': 1, # Scaling factor for metabolic rates at rest

            'F_0': 0, # N, 
            'l_0': 8.9e-3, # m,
            'mass': 3.9e-3, # g, 
            
            # Heat data used for optimisation 
            # 'heat_exp_rec': np.array((2.518854e-02, 6.404227e-02, 6.903081e-02, 7.313249e-02)), # J/F0l0, Fast, recovery heat 
            # 'heat_exp_init': np.array((0.667, 0.616, 0.606,0.606)), # J/F0l0, Fast, initial heat
            'heat_exp_rec': np.array((2.385859e-02, 6.066084e-02, 6.538598e-02, 6.927110e-02)), # J/F0l0, Fast, recovery heat, ASSUME SAME 1/3 REC RATE DURING CONTRACTION
            'heat_exp_init': np.array((6.821729e-01, 6.592259e-01, 6.575630e-01, 6.561958e-01)), # J/F0l0, Fast, initial heat, ASSUME SAME 1/3 REC RATE DURING CONTRACTION
        },

        # Contraction dependent
        # NOTE: k_rest is not used in this implementation
        'k_rest': 0,# 0.0014, # 1/s, Vicini 2000, estimated off of experimental data Blei et al. 1993
        'k_stim': 0.0139,  # 1/s, Vicini 2000, estimated from exp data Blei et al. 1993
        'k_post': 0.9 * 0.0139,  # 1/s, Vicini 2000, estimated from exp data Blei et al. 1993 NOTE: cannot find value for this rate... assume half of stim?

        # Assume constant across all species and muscle fibre-types
        'V_ck_f': 100,# 100, # mM/s, Kushmerick 1998
        'K_b': 1.11, #mM, MacFarland 1994
        'K_ia': 0.135, # mM, MacFarland 1994
        'K_eq': 1.77e2, # ?, Assuming a pH of 7, Lawson 1979
        'K_iq': 3.5, # mM, MacFarland 1994
        'K_ib': 3.9, # mM, MacFarland 1994
        'K_p': 3.8, # mM, MacFarland 1994

        # Energetic constant to predict energetic rates 
        # 'r_rec': 1 / 0.8 * 60e3 # J / mol, Assumes mitochondrial efficiency based on average mouse sol and edl
        'Gatp': 60e3, # J/mol, Free energy of ATP (Barclay 2019)

        # Mechanical parameters (may not be used)
        'k_see': 0, # Unused

        # Q10 values for scaling input experimental data
        'q10_heat': 1,

    
}
# Calculate the maximum isometric forces
for muscle_ in ('SOL', 'EDL'):
    # Maximum isometric force (Assuming fixed max_iso_stress for now)    
    params[muscle_]['F_0'] = params[muscle_]['mass'] / params['rho0'] / params[muscle_]['l_0'] * params['max_iso_stress']
    print(f'{muscle_}: Maximum isometric force: {params[muscle_]["F_0"]}')

# Define the initial energy consumption 
def E_init(t): 
    '''
    Return energy from initial processes in F0l0/s 
    '''
    trampend = 1
    t_start_cycle = params['t_cycle_start']
    t_end_cycle = params['t_cycle_end']

    # Normalize time
    t_cycle_length = 5 # s, Length of the cycle
    t_cycle = t%t_cycle_length

    # ge the initial heat rates 
    dHidt_vec = params[params['muscle']]['heat_exp_init']
    
    if params['muscle'] == 'SOL': 
        cycles = np.array((1,5,15,30)) # Cycle number

        # Assuming recovery maintains the same during the contraction period
        # dHidt_vec = np.array((0.225, 0.176, 0.166, 0.164))  # F_0 l_0 / s, slow, initial 

        # Use an exponential fit 
        def f(x, a, b, c, d): 
            return a * b ** x - c
        popt, _ = curve_fit(f, cycles, dHidt_vec)

        print(f'Constants for the function {popt}')

        # Compute cycle index relative to stimulation start so cycle 1 begins at t_cycle_start.
        cycle_count = np.floor((t - t_start_cycle) / t_cycle_length) + 1
        cycle_count = np.maximum(cycle_count, 1)
        dHidt_vec_full = f(cycle_count, *popt)

        tstimend = 0.8 # s, Length of stimulation (B1995)
    elif params['muscle'] == 'EDL':
        cycles = np.array((1,5,15,30)) # Cycle number

        # Assuming recovery maintains the same during the contraction period
        # dHidt_vec = np.array((0.667, 0.616, 0.606,0.606)) # umol/g, fast 

        # Use an exponential fit 
        def f(x, a, b, c, d): 
            return a * b ** x - c
        popt, _ = curve_fit(f, cycles, dHidt_vec)

        print(f'Constants for the function {popt}')

        # Compute cycle index relative to stimulation start so cycle 1 begins at t_cycle_start.
        cycle_count = np.floor((t - t_start_cycle) / t_cycle_length) + 1
        cycle_count = np.maximum(cycle_count, 1)
        dHidt_vec_full = f(cycle_count, *popt)
        
        tstimend = 0.2 # s, Length of stimulation (B1995)

    # Energy Q10 value
    q10 = params['q10_heat']

    return (dHidt_vec_full * (t_cycle < tstimend)) * (t >= t_start_cycle) * (t <= t_end_cycle) * q10 # F0l0 / s
          
# Initialise the intial energy 
t_vec = np.linspace(params['t_start'], params['t_end'], 100 * params['t_end'])
t_span = (t_vec[0], t_vec[-1]) 
c_atp_0 = params[params['muscle']]['c_atp_0']
E_tot = E_init(t_vec) / params[params['muscle']]['mass'] # Units of F0l0 / g / s

# Compute the scaled initial energy 
e_init_scale = params[params['muscle']]['mass'] # g
E_tot_scaled = E_tot * e_init_scale # Units of F0l0 / s
E_tot_bioenergy_input = E_tot * params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] # W / g

#########
# Define the code for the optimisation 
muscle = params['muscle']


# Define the recovery heat from the experimental data 
cycle = np.array((1,5,15,30)) # Cycle numbers
t_exp = params['t_cycle_start'] +  cycle * 5 # s, Times for the experimental values 

def rec_heat_exp(t, heat_exp_rec): 
    # Use an exponential form consistent with E_init, but increasing with cycle.
    def f_rec(x, a, b, c, d):
        return c - a * b ** x

    popt, _ = curve_fit(f_rec, cycle, heat_exp_rec)
    cycle_count = np.floor((t - params['t_cycle_start']) / 5) + 1
    cycle_count = np.maximum(cycle_count, 1)
    heat_rec_fit = f_rec(cycle_count, *popt)

    return heat_rec_fit * (t >= params['t_cycle_start']) * (t < params['t_cycle_end']) * params['q10_heat']

# Rerun the model with the optimal values 
bioenergetic_model = Bioenergetics(params) 
# Solve the model
sol = bioenergetic_model.solveBioenergetics(t_span, c_atp_0, t_vec, E_tot_bioenergy_input)
# Compute the energetic rates 
q_r_unscaled = bioenergetic_model.computeRecoveryEnergetics(sol.t, sol.y[0,]) # In units of W / g
scale = params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] / params[params['muscle']]['mass']
q_r = q_r_unscaled / scale # In units of F0l0 / s

# # Plot with units in F0l0/s
# # Plot the rates 
# fig, ax = plt.subplots(layout = 'constrained')
# # e_init_scale = (params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] / params[params['muscle']]['mass'])
# # E_tot = E_tot * e_init_scale
# # energy_unit_scaler = params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] / params[params['muscle']]['mass'] * 1e3 # convert from W/F0l0 to mW/g 
energy_unit_scaler = 1 
# ax.plot(t_vec, E_tot_scaled * energy_unit_scaler , label = '$\dot e_{init}$') 
# ax.plot(t_vec, q_r * energy_unit_scaler, label = '$\dot q_r$') 
# ax.plot(t_vec, (E_tot_scaled + q_r) * energy_unit_scaler, label = '$\dot q_r + \dot e_{init}$') 
# ax.legend()
# ax.set_xlabel('Time (s)')
# # ax.set_ylabel('Energy rate ($mW g^{-1}$)')
# ax.set_ylabel('Energy rate ($F_0 l_0 s^{-1}$)')
# Compare model and experimental recovery heat rates with same plots as optimisation 
recovery_heat_exp = rec_heat_exp(sol.t, params[muscle]['heat_exp_rec'])
# Per-cycle means over the same fitting window used in optimisation.
cycle_length = 5.0
fit_mask_plot = (sol.t > params['t_cycle_start'] + cycle_length) * (sol.t < params['t_cycle_end'])
t_fit_plot = sol.t[fit_mask_plot]
model_fit_plot = q_r[fit_mask_plot]
exp_fit_plot = recovery_heat_exp[fit_mask_plot]
cycle_idx_plot = np.floor((t_fit_plot - params['t_cycle_start']) / cycle_length).astype(int)
unique_cycles_plot = np.unique(cycle_idx_plot)
cycle_centers_plot = params['t_cycle_start'] + (unique_cycles_plot + 0.5) * cycle_length
model_cycle_means = np.array([np.mean(model_fit_plot[cycle_idx_plot == cyc]) for cyc in unique_cycles_plot])
exp_cycle_means = np.array([np.mean(exp_fit_plot[cycle_idx_plot == cyc]) for cyc in unique_cycles_plot])
# Plot
fig, ax = plt.subplots(layout = 'constrained', figsize = (7,5))
ax.plot(sol.t, q_r, label='Mod rec heat rate')
ax.plot(sol.t, recovery_heat_exp, label='Exp rec heat rate', linestyle='--')
ax.plot(cycle_centers_plot, model_cycle_means, 's', ms=5, label='Mod cycle mean')
# ax.plot(cycle_centers_plot, exp_cycle_means, 'd', ms=5, label='Exp cycle mean')
cycle_rec = np.array((1, 5, 15, 30))
t_exp_rec = params['t_cycle_start'] + 5 * cycle_rec
ax.plot(t_exp_rec, params[muscle]['heat_exp_rec'] * params['q10_heat'], 'o', label='Experimental data points')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Recovery heat rate ($F_0 l_0 s^{-1}$)')
ax.legend()
ax.grid(True)
fig.savefig('Figures/B1995Opt_' + muscle + '__expdatacomp' + save_id + '.jpg')
fig.savefig('Figures/B1995Opt_' + muscle + '__expdatacomp' + save_id + '.svg')

# Plot the total energy over the cycle
fig, ax = plt.subplots(layout = 'constrained', figsize = (7,5))
ax.plot(t_vec, cumtrapz(E_tot_scaled, t_vec, initial = 0) * energy_unit_scaler, label = '$ e_{init}$') 
ax.plot(t_vec, cumtrapz(q_r, t_vec, initial = 0) * energy_unit_scaler, label = '$ q_r$') 
ax.plot(t_vec, cumtrapz(E_tot_scaled + q_r, t_vec, initial = 0) * energy_unit_scaler, label = '$ q_r + e_{init}$')
ax.legend()
ax.set_xlabel('Time (s)')
# ax.set_ylabel('Energy  ($mJ g^{-1}$)')
ax.set_ylabel('Energy  ($F_0 l_0$)')
fig.savefig('Figures/B1995Opt_' + muscle + '_totalenergy' + save_id + '.jpg')
fig.savefig('Figures/B1995Opt_' + muscle + '_totalenergy' + save_id + '.svg')

# Compare model and experimental recovery heat rates
# recovery_heat_exp = rec_heat_exp(sol.t, params[muscle]['heat_exp_rec'])
# fig, ax = plt.subplots(layout = 'constrained', figsize = (7,5))
# ax.plot(sol.t, q_r, label='Model recovery heat rate')
# ax.plot(sol.t, recovery_heat_exp, label='Experimental recovery heat rate (interp.)', linestyle='--')
# ax.plot(t_exp, params[muscle]['heat_exp_rec'] * params['q10_heat'], 'o', label='Experimental data points')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Recovery heat rate ($F_0 l_0 s^{-1}$)')
# ax.legend()
# ax.grid(True)
# fig.savefig('Figures/B1995Opt_' + muscle + '_expdatacomp' + save_id + '.jpg')
# fig.savefig('Figures/B1995Opt_' + muscle + '_expdatacomp' + save_id + '.svg')

# Compute the time constant from an exponential fit on post-contraction decay
energy_unit_scaler = params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] / params[params['muscle']]['mass'] * 1e3 # convert from W/F0l0 to mW/g
total_energy_rate = (E_tot_scaled + q_r) * energy_unit_scaler

# Fit only after stimulation has ended
mask = t_vec >= params['t_cycle_end']
t_decay = t_vec[mask]
y_decay = total_energy_rate[mask]
t_rel = t_decay - t_decay[0]

def exp_decay(t, y_inf, A, tau):
    return y_inf + A * np.exp(-t / tau)

# Initial guesses: asymptote from tail mean, amplitude from first point, tau as 20 s
tail_n = min(500, len(y_decay))
y_inf_guess = float(np.mean(y_decay[-tail_n:]))
A_guess = float(y_decay[0] - y_inf_guess)
tau_guess = 20.0

p0 = (y_inf_guess, A_guess, tau_guess)
bounds = ([-np.inf, -np.inf, 1e-9], [np.inf, np.inf, np.inf])

popt, _ = curve_fit(exp_decay, t_rel, y_decay, p0=p0, bounds=bounds, maxfev=20000)
y_inf_fit, A_fit, tau_fit = popt

print(f'Fitted time constant (tau) = {tau_fit:.3f} s')

# Plot fitted exponential against the decay data
fig, ax = plt.subplots(layout='constrained')
ax.plot(t_rel, y_decay, label='Decay data')
ax.plot(t_rel, exp_decay(t_rel, *popt), '--', label=f'Exp fit (tau = {tau_fit:.2f} s)')
ax.set_xlabel('Time since end of stimulation (s)')
ax.set_ylabel('Total energy rate ($mW\,g^{-1}$)')
# ax.set_title(f'Exponential Decay Fit ({muscle})')
ax.legend()
ax.grid(True)
# fig.savefig('Figures/B1995Opt_' + muscle + '_expdatacomp.jpg')
# fig.savefig('Figures/B1995Opt_' + muscle + '_expdatacomp.svg')

# Compute the thermodynamic efficiency based on substrates + thermodynamics theory 
# TODO: FIX THIS CALCULATION
n_atp = 38 # number of atp
Gatp = 60e3 # J/mol 
# r_rec is in units of J / mol
q_r_totalheat = cumtrapz(q_r, t_vec, initial = 0) * energy_unit_scaler
r_rec_J = q_r_totalheat[-1]
efficiency = n_atp * Gatp / (params[muscle]['r_rec'] + n_atp * Gatp)
print(f'    efficiency: {efficiency}')


plt.show()
