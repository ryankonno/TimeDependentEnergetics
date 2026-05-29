'''
Code to determine the bioenergetics model recovery parameters.

Ryan Konno
r.konno@uq.edu.au
The University of Queensland 
'''

# Import 
import numpy as np 
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt 
import lib.plot_style
from lib.recursive_merge import recursive_merge

import sys 
sys.path.append('./')

from Models.BioenergeticsModel import Bioenergetics

# Import parameters 
from parameters_muscle import params as params_muscle

# Define parameter values for the protocol
params_protocol = {
    # Time parameters for setting up the protocol 
    't_start': 0, # s
    't_end': 300, # s
    't_cycle_start': 10, # s
    't_cycle_end': 10 + 150, # s 

    # Define a q10 value to scale heat (not implemented)
    'q10_heat': 1, 

    'muscle': 'SOL', # Specify muscle parameters to be used in simulation
    
        # Mouse data 
        'SOL': {
            # Stimulation time s 
            't_stim_end': 0.8, # s, Length of stimulation (B1995)

            # Values from Barclay et al. 1995
            'F_0': 0, # N, 
            'l_0':  9.5e-3, # m,    
            'mass': 1.99e-3, # g, 

            # Heat data used for optimisation 
            'heat_exp_rec': np.array((5.339988e-03, 5.726164e-02, 7.235249e-02, 7.009909e-02)), # J/F0l0, Slow, recovery heat 
            'heat_exp_init': np.array((0.225, 0.176, 0.166, 0.164)),  # J/F0l0, Slow, initial heat, ASSUME SAME REC RATE DURING CONTRACTION
            # 'heat_exp_rec': np.array((4.478470e-03, 4.802343e-02, 6.067962e-02, 5.878977e-02)), # J/F0l0, Slow, recovery heat, ASSUME SAME 1/3 REC RATE DURING CONTRACTION
            # 'heat_exp_init': np.array((2.362291e-01, 2.190962e-01, 2.193518e-01, 2.148170e-01)),  # J/F0l0, Slow, initial heat, ASSUME SAME 1/3 REC RATE DURING CONTRACTION
            # 'heat_exp_rec': np.array((4.693850e-03, 5.033298e-02, 6.359784e-02, 6.161710e-02)), # J/F0l0, Slow, recovery heat, ASSUME SAME 1/2 REC RATE DURING CONTRACTION
            # 'heat_exp_init': np.array((2.353391e-01, 2.095526e-01, 2.072931e-01, 2.031338e-01)),  # J/F0l0, Slow, initial heat, ASSUME SAME 1/2 REC RATE DURING CONTRACTION
        }, 
        'EDL': {            
            # Stimulation time s 
            't_stim_end': 0.2, # s, Length of stimulation (B1995)
            
            # Values from Barclay et al. 1995
            'F_0': 0, # N, 
            'l_0': 8.9e-3, # m,
            'mass': 3.9e-3, # g, 

            # Heat data used for optimisation 
            'heat_exp_rec': np.array((2.518854e-02, 6.404227e-02, 6.903081e-02, 7.313249e-02)), # J/F0l0, Fast, recovery heat 
            'heat_exp_init': np.array((0.667, 0.616, 0.606,0.606)), # J/F0l0, Fast, initial heat
            # 'heat_exp_rec': np.array((2.385859e-02, 6.066084e-02, 6.538598e-02, 6.927110e-02)), # J/F0l0, Fast, recovery heat, ASSUME SAME 1/3 REC RATE DURING CONTRACTION
            # 'heat_exp_init': np.array((6.821729e-01, 6.592259e-01, 6.575630e-01, 6.561958e-01)), # J/F0l0, Fast, initial heat, ASSUME SAME 1/3 REC RATE DURING CONTRACTION
            # 'heat_exp_rec': np.array((2.419108e-02, 6.150620e-02, 6.629719e-02, 7.023644e-02)), # J/F0l0, Fast, recovery heat, ASSUME SAME 1/2 REC RATE DURING CONTRACTION
            # 'heat_exp_init': np.array((6.779748e-01, 6.485522e-01, 6.460579e-01, 6.440071e-01)), # J/F0l0, Fast, initial heat, ASSUME SAME 1/2 REC RATE DURING CONTRACTION
        },
}

# Combine the dictionaries
params = recursive_merge(params_muscle, params_protocol)

# Compute parameter values not in the dictionaries
for muscle in ('SOL', 'EDL'):
    # Maximum isometric force
    params[muscle]['F_0'] = (
        params[muscle]['mass'] / params['rho0'] /
        params[muscle]['l_0'] *
        params[muscle]['max_iso_stress']
    )
    print(f'{muscle}: Maximum isometric force: {params[muscle]["F_0"]}')


# Define initial energy consumption (determined based on experimental protocol)
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

    # Get the initial heat rates 
    dHidt_vec = params[params['muscle']]['heat_exp_init']
    t_stim_end = params[params['muscle']]['t_stim_end']

    cycles = np.array((1,5,15,30)) # Cycle number

    # Use an exponential fit 
    def f(x, a, b, c, d): 
        return a * b ** x - c
    
    popt, _ = curve_fit(f, cycles, dHidt_vec)

    print(f'Constants for the function {popt}')

    # Compute cycle index relative to stimulation start so cycle 1 begins at t_cycle_start.
    cycle_count = np.floor((t - t_start_cycle) / t_cycle_length) + 1
    cycle_count = np.maximum(cycle_count, 1)
    dHidt_vec_full = f(cycle_count, *popt)

    # Energy Q10 value (See Glens power dataset conversion (used in Konno et al., 2025))
    q10 = params['q10_heat']
    
    return (dHidt_vec_full * (t_cycle < t_stim_end)) * (t >= t_start_cycle) * (t <= t_end_cycle) * q10 #  F0l0 / s

# Plot to verify input energy usage
fig,ax = plt.subplots(layout = 'constrained')
t_vec_plot = np.linspace(0,150, 1000)
cycles = np.array((1,5,15,30))
ax.plot(t_vec_plot, E_init(t_vec_plot), label = 'Interpolation', lw = 2)
ax.plot(params['t_cycle_start'] + 5 * cycles, params[params['muscle']]['heat_exp_init'] * params['q10_heat'], label = 'Raw data (B1995)', ls = 'None', marker = '.', ms = 10)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Recovery heat ($F_0 l_0 s^{-1}$)')
ax.legend()
# fig.savefig('Figures/2_optimisation_recovery_energy_initial_heat_fit.jpg')
# fig.savefig('Figures/2_optimisation_recovery_energy_initial_heat_fit.svg')
# # plt.savefig('./Figures/B1995QiFit_SOL.pdf')
plt.show()

#Function to get the recovery heat 
def rec_heat_exp(t, heat_exp_rec): 
    cycle = np.array((1,5,15,30)) # Cycle numbers

    # Use an exponential form consistent with E_init, but increasing with cycle.
    def f_rec(x, a, b, c, d):
        return c - a * b ** x

    popt, _ = curve_fit(f_rec, cycle, heat_exp_rec)
    cycle_count = np.floor((t - params['t_cycle_start']) / 5) + 1
    cycle_count = np.maximum(cycle_count, 1)
    heat_rec_fit = f_rec(cycle_count, *popt)

    return heat_rec_fit * (t >= params['t_cycle_start']) * (t < params['t_cycle_end']) * params['q10_heat'] # Units of F0l0/s

# Plot the recovery heat to verify
cycle_rec = np.array((1, 5, 15, 30))
t_exp_rec = params['t_cycle_start'] + 5 * cycle_rec
fig, ax = plt.subplots(layout='constrained')
ax.plot(t_vec_plot, rec_heat_exp(t_vec_plot, params[params['muscle']]['heat_exp_rec']), label='Exponential fit', lw=2,)
ax.plot(t_exp_rec, params[params['muscle']]['heat_exp_rec'] * params['q10_heat'], label='Raw data (B1995)', ls='None', marker='.', ms=10,)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Recovery heat ($F_0 l_0 s^{-1}$)')
ax.legend()
# fig.savefig('Figures/2_optimisation_recovery_energy_recovery_heat_fit.jpg')
# fig.savefig('Figures/2_optimisation_recovery_energy_recovery_heat_fit.svg')
plt.show()
          
# Initialise the intial energy 
t_vec = np.linspace(params['t_start'], params['t_end'], 100 * params['t_end'])
t_span = (t_vec[0], t_vec[-1]) 
E_tot = E_init(t_vec) / params[params['muscle']]['mass'] # Units of  F0 l0 / g / s

# Compute the scaled initial energy 
e_init_scale = params[params['muscle']]['mass'] # g
E_tot_scaled = E_tot * e_init_scale # Units of F0l0 / s
E_tot_bioenergy_input = E_tot * params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] # W / g

# Initialise atp concentration 
c_atp_0 = params[params['muscle']]['c_atp_0']

# Define the code for the optimisation 
muscle = params['muscle']

# Optimisation function
def f_opt(x):
    ####
    # Define the bioenergetics model 
    model = Bioenergetics(params)
    model.r_rec = x[0] * 1e6
    model.nh = x[1]
    model.V_max_oxphos = x[2]

    #####
    # Solve the ode 
    sol = model.solveBioenergetics(t_span, c_atp_0, t_vec, E_tot_bioenergy_input) # Note E_tot here in W / g
    
    # Compute the recovery 
    recovery_heat_model_ = model.computeRecoveryEnergetics(sol.t, sol.y[0,]) # W / g / s 

    # Scale the recovery heat 
    # scale_factor = params[params['muscle']]['F_0'] * params[params['muscle']]['l_0']
    recovery_heat_model = recovery_heat_model_ # Units of W / g / s

    # compute the recovery heat from the experiment 
    scale_factor_exp = params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] / params[params['muscle']]['mass']
    recovery_heat_exp = rec_heat_exp(sol.t, params[muscle]['heat_exp_rec'])  * scale_factor_exp # Units of W / g / s

    # Fit cycle-averaged recovery heat (one mean value per cycle) instead of
    # using the full pointwise trace.
    cycle_length = 5.0
    fit_mask = (sol.t > params['t_cycle_start'] + cycle_length) * (sol.t < params['t_cycle_end'])
    t_fit = sol.t[fit_mask]
    model_fit = recovery_heat_model[fit_mask]
    exp_fit = recovery_heat_exp[fit_mask]
    cycle_idx = np.floor((t_fit - params['t_cycle_start']) / cycle_length).astype(int)
    unique_cycles = np.unique(cycle_idx)
    recovery_heat_model_crop = np.array([np.mean(model_fit[cycle_idx == cyc]) for cyc in unique_cycles])
    recovery_heat_exp_crop = np.array([np.mean(exp_fit[cycle_idx == cyc]) for cyc in unique_cycles])
    
    ################################
    # Opt 1: just minimise over initial time period 
    error = np.linalg.norm(recovery_heat_model_crop - recovery_heat_exp_crop) / np.linalg.norm(recovery_heat_exp_crop)

    ################################
    # Opt 2: minimise over the initial time period and match the ratio of initial to recovery energy post-contraction 
    # error_init_per = np.linalg.norm(recovery_heat_model_crop - recovery_heat_exp_crop) / np.linalg.norm(recovery_heat_exp_crop)
    # E_init_final = np.trapz(E_tot_scaled, t_vec) # Get final initial energy 
    # E_rec_final = np.trapz(recovery_heat_model, t_vec) # Get final recovery energy 
    # error_ratio_final = np.linalg.norm(E_init_final /  E_rec_final - 1) / np.linalg.norm(E_init_final)
    # w = 1 # Weighting (not sure if needed)
    # error = error_init_per + w * error_ratio_final
    # print(f'______________________________')
    # print(f'error_init =  {error_init_per}')
    # print(f'error_ratio = {error_ratio_final}')

    # # ################################
    # # # Opt 3: add in a condition to minimise the derivative
    # error_init_per = np.linalg.norm(recovery_heat_model_crop - recovery_heat_exp_crop) / np.linalg.norm(recovery_heat_exp_crop)
    # E_init_final = np.trapz(E_tot_scaled, t_vec) # Get final initial energy 
    # E_rec_final = np.trapz(recovery_heat_model, t_vec) # Get final recovery energy 
    # error_ratio_final = np.linalg.norm(E_init_final /  E_rec_final - 1) / np.linalg.norm(E_init_final)
    # E_rec_deriv = np.diff(recovery_heat_model) / np.diff(t_vec)
    # error_deriv_final = np.abs(E_rec_deriv[-1])
    # w1 = x[4] * 10 # Weighting (not sure if needed)
    # w2 = x[5] * 10# Weighting for derivative
    # error = error_init_per + w1 * error_ratio_final + w2 * error_deriv_final

    # print(f'______________________________')
    # print(f'error_init =  {error_init_per}')    
    # print(f'error_ratio = {w1 * error_ratio_final}')
    # print(f'error_deriv = {w2 * error_deriv_final}')

    # Error, with accounting for the total heat post simulation (should have ratio between initial and recovery heat \approx 1)

    return error 

# Define a callback function to output regular results 
global iter 
iter = 0
def callback_fun(xk):
    global iter 
    iter += 1

    # Callkback for opt 1:
    print(f'iter: {iter}, r_rec = {xk[0]}, nh = {xk[1]}, V_max_oxphos = {xk[2]}')
    
    return 0 

# Opt r_rec, nh, and V_max 

# Define bounds and initial conditions
x0 = (params[muscle]['r_rec'] / 1e6, 0.8, 1.5)
bounds_ = ((0.01,4), (0.01, 4), (0.5,10))

# Perform the optimisation
opt_res = minimize(f_opt, x0, callback= callback_fun, bounds = bounds_,  options = {'maxiter': 500, 'disp': True}, method='Nelder-Mead')

# Print out the optimal solution 
x = opt_res.x 
print(f'r_rec = {x[0]}, nh = {x[1]}, V_max_oxphos = {x[2]}')
params[muscle]['r_rec'], params[muscle]['nh'], params[muscle]['V_max_oxphos'] = x[0] * 1e6, x[1], x[2]

##################################
# Rerun the model with the optimal values 
bioenergetic_model = Bioenergetics(params) 
# Solve the model
sol = bioenergetic_model.solveBioenergetics(t_span, c_atp_0, t_vec, E_tot_bioenergy_input)
# Compute the energetic rates 
q_r_unscaled = bioenergetic_model.computeRecoveryEnergetics(sol.t, sol.y[0,]) # In units of W / g 
scale = params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] / params[params['muscle']]['mass']
q_r = q_r_unscaled / scale # F0l0 / s

##################################
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
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(sol.t, q_r, label='Model recovery heat rate')
ax.plot(sol.t, recovery_heat_exp, label='Experimental recovery heat rate (interp.)', linestyle='--')
ax.plot(cycle_centers_plot, model_cycle_means, 's', ms=5, label='Model cycle mean')
ax.plot(cycle_centers_plot, exp_cycle_means, 'd', ms=5, label='Experimental cycle mean')
ax.plot(t_exp_rec, params[muscle]['heat_exp_rec'] * params['q10_heat'], 'o', label='Experimental data points')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Recovery heat rate ($F_0 l_0 s^{-1}$)')
ax.legend()
ax.grid(True)
# fig.savefig('Figures/2_optimisation_recovery_energy_cycle_mean_comparison.jpg')
# fig.savefig('Figures/2_optimisation_recovery_energy_cycle_mean_comparison.svg')

# Compare model and experimental recovery heat rates
recovery_heat_exp = rec_heat_exp(sol.t, params[muscle]['heat_exp_rec'])
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(sol.t, q_r, label='Model recovery heat rate')
ax.plot(sol.t, recovery_heat_exp, label='Experimental recovery heat rate (interp.)', linestyle='--')
ax.plot(t_exp_rec, params[muscle]['heat_exp_rec'] * params['q10_heat'], 'o', label='Experimental data points')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Recovery heat rate ($F_0 l_0 s^{-1}$)')
ax.legend()
ax.grid(True)
# fig.savefig('Figures/2_optimisation_recovery_energy_recovery_rate_comparison.jpg')
# fig.savefig('Figures/2_optimisation_recovery_energy_recovery_rate_comparison.svg')


# Plot with units in F0l0/s
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, (E_tot_scaled + q_r), label = '$\dot q_r + \dot e_{init}$') 
ax.plot(t_vec, E_tot_scaled , label = '$\dot e_{init}$') 
ax.plot(t_vec, q_r, label = '$\dot q_r$') 
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Energy rate ($F_0 l_0 s^{-1}$)')
# fig.savefig('Figures/2_optimisation_recovery_energy_rate_components.jpg')
# fig.savefig('Figures/2_optimisation_recovery_energy_rate_components.svg')

# Plot the total energy over the cycle
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, cumulative_trapezoid(E_tot_scaled + q_r, t_vec, initial = 0), label = '$ q_r + e_{init}$') 
ax.plot(t_vec, cumulative_trapezoid(E_tot_scaled, t_vec, initial = 0), label = '$ e_{init}$') 
ax.plot(t_vec, cumulative_trapezoid(q_r, t_vec, initial = 0), label = '$ q_r$') 
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Energy  ($F_0 l_0$)')
# fig.savefig('Figures/2_optimisation_recovery_energy_integral_components.jpg')
# fig.savefig('Figures/2_optimisation_recovery_energy_integral_components.svg')
plt.show()

# Print the ratio between initial and recovery heat 
q_r_cum = cumulative_trapezoid(q_r, t_vec, initial = 0)
e_init_cum = cumulative_trapezoid(E_tot_scaled, t_vec, initial = 0)
dt = t_vec[1] - t_vec[0]
print(f'Ratio in initial and recovery heat: {e_init_cum[int(params["t_cycle_end"]/dt)] / q_r_cum[int(params["t_cycle_end"]/dt)]}')