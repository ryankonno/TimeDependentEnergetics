'''
Fit the recovery heat parameter to the B1995 dataset. Here we also model the initial processes

Ryan Konno
School of Biomedical Sciences 
The University of Queensland
'''

# Import 
import numpy as np 
from scipy.integrate import cumtrapz
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 14
import matplotlib.cm as cmap
import sys 
sys.path.append('./')

from Models.BioenergeticsSimple import Bioenergetics
from scipy.interpolate import PchipInterpolator

params = {
    # Time parameters for setting up the protocol 
    't_start': 0, # s
    't_end': 300, # s
    't_cycle_start': 10, # s
    't_cycle_end': 10 + 150, # s 

    # General muscle parameters
    'rho0':  1e6,    # g/m^3, Density of muscle
    'max_iso_stress': 2.5e5, # N/m^2, Maximum isometric stress of the muscle

    'muscle': 'SOL', # Specify muscle parameters to be used in simulation

        # Mouse data 
        'SOL': {
            # Slow data
            'c_c_tot': 25.9, # mM, Kushmerick et al. 1992 
            'c_atp_0': 3.3, # mM,  Kushmerick et al. 1992 
            'c_pcr_0': 11.4, # mM,  Kushmerick et al. 1992 
            # Fast data 
            # 'c_c_tot': 29.5, # mM, Kushmerick et al. 1992 
            # 'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992 
            # 'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992 

            'V_max_oxphos':  1.88, # mM/s, Vicini 2000... TBD

            'gamma': 1, # Scaling factor for metabolic rates at rest

            # May need to tune these parameters...
            'K_adp': 0.058, # mM, Vicini 2000.... TBD (may need to optimise for this parameter)
            'nh': 2.57, # unitless, V/Icini 2000, .... TBD (may need to optimise for this parameter)

            # Values from Barclay et al. 1995
            'F_0': 0, # N, 
            'l_0':  9.5e-3, # m,    
            'mass': 1.99e-3, # g, 

            # Exsperimental parameters 
            'stim_freq': 200
           
            'r_rec': 0.25e6, # J / mol, Optimised value
            
            # Heat data used for optimisation 
            'heat_exp_rec': np.array((5.339988e-03, 5.726164e-02, 7.235249e-02, 7.009909e-02)), # J/F0l0, Slow, recovery heat 
            # 'heat_exp_rec': np.array((5.339988e-03, 5.726164e-02, 7.235249e-02, 7.009909e-02)), # J/F0l0, Slow, recovery heat 
            'heat_exp_init': np.array((0.225, 0.176, 0.166, 0.164)),  # J/F0l0, Slow, initial heat 

            # Activation model parameters 
            "Tau_1": 0.3, # Assume constant value from MCL (2023)
            # "Tau_2": 0.256, # Scaling based on MCL (2023)
            "Tau_2": 0.02,
            "K": 0.1025,
            "n": 4, # Hill coefficient for act model
            
            # Mechanical parameters 
            'dedt_ce_max': 5, 
            'kappa': 0.18,

            # Initial energetics model 
            'r_am': 0.4599, # W/F_0/l_0, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_sl': 0.2958, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

        }, 
        'EDL': { 
            'c_c_tot': 29.5, # mM, Kushmerick et al. 1992 
            'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992 
            'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992 
            'Pi_0': 0, # mM,  Kushmerick et al. 1992 

            'V_max_oxphos': 1.88/2, # mM/s, Vicini 2000... TBD
            'gamma': 1, # Scaling factor for metabolic rates at rest

            # May need to tune these parameters...
            'K_adp': 0.058, # mM, Vicini 2000.... TBD (may need to optimise for this parameter)
            'nh': 2.57, # unitless, VIcini 2000, .... TBD (may need to optimise for this parameter)

            # Values from Barclay et al. 1995
            'F_0': 0, # N, 
            'l_0': 8.9e-3, # m,
            'mass': 3.9e-3, # g, 
                 
            'r_rec': 0.25e6, # J / mol, Optimised value
            
            # Heat data used for optimisation 
            'heat_exp_rec': np.array((2.518854e-02, 6.404227e-02, 6.903081e-02, 7.313249e-02)), # J/F0l0, Fast, recovery heat 
            'heat_exp_init': np.array((0.667, 0.616, 0.606,0.606)), # J/F0l0, Fast, initial heat

            # Activation model parameters 
            "Tau_1": 0.0422, # Very little change between fibre type - assume constant (BH, 2003)
            # "Tau_2": 0.256/2, # Decay constant for fast twitch Fibres assuming 1/2 rate (Baylor and Hollingworth, 2003)
            "Tau_2": 0.1/2, # Decay constant for fast twitch Fibres assuming 1/2 rate (Baylor and Hollingworth, 2003)
            "K": 0.1025,
            "n": 2, # Hill coefficient for activation model

            # Mechanical parameters 
            'dedt_ce_max': 10, 
            'kappa': 0.29,

            # Energetics model 
            'r_am': 1.0711, # W/F_0/l_0, Maximum heat rate of isometric contraction (fast-type fibre)
            'r_sl': 0.7792, # W/F_0/l_0, Maximum shortening heat rate (fast-type fibre)

        },
        
        # 'c_c_tot': 42, # mM, Harris 1974
        # 'c_atp_0': 8.2, # mM, Harris 1974
        # 'c_pcr_0': 32, # mM, Approximate value Vicini 2000
        # # 'c_c_tot': 20, # mM, Grassi 1998
        # # 'c_atp_0': 6.5, # mM, Grassi 1998
        # 'Pi_0': 3.183, # mM, Vicini 2000

        # 'V_max_oxphos': 0.5, # mM/s, Vicini 2000... TBD (may need to optimise for this parameter)
        # 'V_max_oxphos': 14.8 / 60, # mM/s, Vicini 2000... TBD (may need to optimise for this parameter)

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

        # Q10 values for scaling input experimetnal data 
        'q10_heat': 4,

    
}

# Calculate the maximum isometric forces
for muscle_ in ('SOL', 'EDL'):
    # Maximum isometric force (Assuming fixed max_iso_stress for now)    
    params[muscle_]['F_0'] = params[muscle_]['mass'] / params['rho0'] / params[muscle_]['l_0'] * params['max_iso_stress']
    print(f'{muscle_}: Maximum isometric force: {params[muscle_]["F_0"]}')


# Define initial energy consumption 
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

    if params['muscle'] == 'SOL': 
        cycles = np.array((1,5,15,30)) # Cycle number

        # Assuming recovery maintains the same during the contraction period
        dHidt_vec = np.array((0.225, 0.176, 0.166, 0.164))  # F_0 l_0 / s, slow, initial 
        # dATPdt_vec = np.array((5.339988e-03, 5.726164e-02, 7.235249e-02, 7.009909e-02))   # umol/g, slow, recovery

        # Use an exponential fit 
        def f(x, a, b, c, d): 
            # return a * np.exp( b* x - c) + d
            return a * b ** x - c
            # return 0.1 * a * b **(  x**d) - c
        popt, _ = curve_fit(f, cycles, dHidt_vec)

        print(f'Constants for the function {popt}')

        # Compute cycle index relative to stimulation start so cycle 1 begins at t_cycle_start.
        cycle_count = np.floor((t - t_start_cycle) / t_cycle_length) + 1
        cycle_count = np.maximum(cycle_count, 1)
        dHidt_vec_full = f(cycle_count, *popt)

        tstimend = 0.8 # s, Length of stimulation (B1995)

    elif params['muscle'] == 'EDL':
        cycles = np.array((1,5,15,30)) # Cycle number
        #____
        # Assuming recovery maintains the same during the contraction period
        dHidt_vec = np.array((0.667, 0.616, 0.606,0.606)) # umol/g, fast 
        # dATPdt_vec = np.array((2.518854e-02, 6.404227e-02, 6.903081e-02, 7.313249e-02)) * F_0 * l_0 / mass  / G_atp * 10**6 # umol/g, fast, recovery

        # Use an exponential fit 
        def f(x, a, b, c, d): 
            # return a * np.exp( b* x - c) + d
            return a * b ** x - c
            # return 0.1 * a * b **(  x**d) - c
        popt, _ = curve_fit(f, cycles, dHidt_vec)

        print(f'Constants for the function {popt}')


        # Compute cycle index relative to stimulation start so cycle 1 begins at t_cycle_start.
        cycle_count = np.floor((t - t_start_cycle) / t_cycle_length) + 1
        cycle_count = np.maximum(cycle_count, 1)
        dHidt_vec_full = f(cycle_count, *popt)

        tstimend = 0.2 # s, Length of stimulation (B1995)

    # Energy Q10 value (See Glens power dataset conversion (used in Konno et al., 2025))
    q10 = params['q10_heat']
    
    return (dHidt_vec_full * (t_cycle < tstimend)) * (t >= t_start_cycle) * (t <= t_end_cycle) * q10 #  F0l0 / s

# Plot to verify input energy usage
fig,ax = plt.subplots(layout = 'constrained')
t_vec_plot = np.linspace(0,150, 500)
cycles = np.array((1,5,15,30))
ax.plot(t_vec_plot, E_init(t_vec_plot), label = 'Interpolation', lw = 2)
ax.plot(params['t_cycle_start'] + 5 * cycles, params[params['muscle']]['heat_exp_init'] * params['q10_heat'], label = 'Raw data (B1995)', ls = 'None', marker = '.', ms = 10)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Recovery heat ($F_0 l_0 s^{-1}$)')
ax.legend()
# plt.savefig('./Figures/B1995QiFit_SOL.pdf')
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

#########
# Define the code for the optimisation 
muscle = params['muscle']

# Optimisation function
def f_opt(x):
    ####
    # Define the bioenergetics model 
    model = Bioenergetics(params)
    model.r_rec = x[0] * 1e6
    # model.nh = x[1]
    # model.K_adp = x[2]
    # model.V_max_oxphos = x[3]
    # print(f'r_rec = {model.r_rec}, nh = {model.nh}, K_adp = {model.K_adp}, V_max_oxphos = {model.V_max_oxphos}')

    #####
    # Solve the ode 
    sol = model.solveBioenergetics(t_span, c_atp_0, t_vec, E_tot_bioenergy_input) # Note E_tot here in W / g

    # c_atp, c_pcr = sol.y  # Transpose to get individual variables

    # Compute the recovery 
    recovery_heat_model_ = model.computeRecoveryEnergetics(sol.t, sol.y[0,]) # W / g / s 

    # Scale the recovery heat 
    # scale_factor = params[params['muscle']]['F_0'] * params[params['muscle']]['l_0']
    recovery_heat_model = recovery_heat_model_ # Units of W / g / s

    # compute the recovery heat from the experiment 
    scale_factor_exp = params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] / params[params['muscle']]['mass']
    recovery_heat_exp = rec_heat_exp(sol.t, params[muscle]['heat_exp_rec'])  * scale_factor_exp # Units of W / g / s

    # only fit over the times from t_cycle_start to t_cycle_end 
    recovery_heat_model_crop = recovery_heat_model[(sol.t > params['t_cycle_start'] + 5) * (sol.t < params['t_cycle_end'])]
    recovery_heat_exp_crop = recovery_heat_exp[(sol.t > params['t_cycle_start']+ 5) * (sol.t < params['t_cycle_end'])]
    
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
    # # # Opt 3: TODO add in a condition to minimise the derivative
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
    # xk = opt_res.x 

    # Callkback for opt 1:
    # print(f'iter: {iter}, r_rec = {xk[0]}, nh = {xk[1]}, K_adp = {xk[2]}, V_max_oxphos = {xk[3]}')
    print(f'iter: {iter}, r_rec = {xk[0]}')
    

    # Callback for opt 3:
    # print(f'iter: {iter}, r_rec = {xk[0]}, nh = {xk[1]}, K_adp = {xk[2]}, V_max_oxphos = {xk[3]}, w0 = {xk[4]}, w1 = {xk[5]}')

    # OTHER 
    # print(f'iter: {iter}, r_rec = {xk[0]}')
    return 0 

# Define initial conditions and bounds for parameters 

# # For opt 1
# x0 = (params[muscle]['r_rec'] / 1e6, params[muscle]['nh'], params[muscle]['K_adp'], params[muscle]['V_max_oxphos'])
# bounds_ = ((0.01,4), (0.5,5), (0.001,0.4), (0.001,5))
# # Perform the optimisation
# opt_res = minimize(f_opt, x0, callback= callback_fun, bounds = bounds_,  options = {'maxiter': 500, 'disp': True}, method='Nelder-Mead')
# # Print out the optimal solution 
# x = opt_res.x 
# print(f'r_rec = {x[0]}, nh = {x[1]}, K_adp = {x[2]}, V_max_oxphos = {x[3]}')
# # update the parameter values , note multiplication factors
# params[muscle]['r_rec'], params[muscle]['nh'], params[muscle]['K_adp'], params[muscle]['V_max_oxphos'] = x[0] * 1e6, x[1], x[2], x[3]

# Opt1: only optimise the r_rec variable 
x0 = (params[muscle]['r_rec'] / 1e6)
bounds_ = ((0.01,4),)
# Perform the optimisation
opt_res = minimize(f_opt, x0, callback= callback_fun, bounds = bounds_,  options = {'maxiter': 500, 'disp': True}, method='Nelder-Mead')
# Print out the optimal solution 
x = opt_res.x 
print(f'r_rec = {x[0]}')
# update the parameter values , note multiplication factors
params[muscle]['r_rec'] = x[0] * 1e6

# For opt 3
# w0_0, w1_0 = 1,1
# x0 = (params[muscle]['r_rec'] / 1e6, params[muscle]['nh'], params[muscle]['K_adp'], params[muscle]['V_max_oxphos'], w0_0, w1_0)
# bounds_ = ((0.01,4), (0.5,5), (0.001,0.4), (0.001,5), (0.001,5), (0.1,100))
# # Perform the optimisation
# opt_res = minimize(f_opt, x0, callback= callback_fun, bounds = bounds_,  options = {'maxiter': 500, 'disp': True}, method='Nelder-Mead')
# # Print out the optimal solution 
# x = opt_res.x 
# print(f'r_rec = {x[0]}, nh = {x[1]}, K_adp = {x[2]}, V_max_oxphos = {x[3]}')
# # update the parameter values , note multiplication factors
# params[muscle]['r_rec'], params[muscle]['nh'], params[muscle]['K_adp'], params[muscle]['V_max_oxphos'] = x[0] * 1e6, x[1], x[2], x[3] 

##################################
# Rerun the model with the optimal values 
bioenergetic_model = Bioenergetics(params) 
# Solve the model
sol = bioenergetic_model.solveBioenergetics(t_span, c_atp_0, t_vec, E_tot_bioenergy_input)
# Compute the energetic rates 
q_r_unscaled = bioenergetic_model.computeRecoveryEnergetics(sol.t, sol.y[0,]) # In units of W / g / s
scale = params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] / params[params['muscle']]['mass']
q_r = q_r_unscaled / scale # F0l0 / s

##################################
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

# Plot with units in F0l0/s
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, (E_tot_scaled + q_r), label = '$\dot q_r + \dot e_{init}$') 
ax.plot(t_vec, E_tot_scaled , label = '$\dot e_{init}$') 
ax.plot(t_vec, q_r, label = '$\dot q_r$') 
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Energy rate ($F_0 l_0 s^{-1}$)')

# Plot the total energy over the cycle
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, cumtrapz(E_tot_scaled + q_r, t_vec, initial = 0), label = '$ q_r + e_{init}$') 
ax.plot(t_vec, cumtrapz(E_tot_scaled, t_vec, initial = 0), label = '$ e_{init}$') 
ax.plot(t_vec, cumtrapz(q_r, t_vec, initial = 0), label = '$ q_r$') 
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Energy  ($F_0 l_0$)')
plt.show()

# Compute the time constant 
energy_unit_scaler = params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] / params[params['muscle']]['mass'] * 1e3 # convert from W/F0l0 to mW/g 
total_energy_rate = (E_tot_scaled + q_r) * energy_unit_scaler
total_energy_rate_crop = total_energy_rate[t_vec > 100]

# Get the index of the time constant
idx = np.argmin(np.abs(total_energy_rate_crop[0] * 1/np.e - total_energy_rate_crop)) 
t_vec_crop = t_vec[t_vec >100]
t_vec_crop = t_vec_crop - t_vec_crop[0]

# Get the time constant
print(f'Time constant = {t_vec_crop[idx]}s')
