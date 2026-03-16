'''
This is a code to simulate bioenergetics of muscle contraction
Within this code the modelling is limited to the amount of PCr within the muscle, 
based on a prescribed use of ATP. 
The full contraction is not simulated within this code 

This version will optimise to the B1995 dataset while ensuring the correct ratio of initial to recovery heat after sufficient time

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

params = {
    # Time parameters for setting up the protocol 
    't_start': 0, # s
    't_end': 300, # s
    't_cycle_start': 10, # s
    't_cycle_end': 150, # s 
    'cycle_length': 1, # s, Defines the length of the cycle (used to set frequency of the contractions)
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
            # 'Pi_0': 6, # mM,  Kushmerick et al. 1992 
            'V_max_oxphos':  1.88, # mM/s, Vicini 2000... TBD
            # 'V_max_oxphos':  0.5, # mM/s, Vicini 2000... TBD
            'atp_peak': 0.25,# 0.213, # mM/s Peak atp rate calculated based on initial heat rate and enthalpy of ATP from Phillips et al. 1993

            'gamma': 0.5, # Scaling factor for metabolic rates at rest

            # May need to tune these parameters...
            'K_adp': 0.058, # mM, Vicini 2000.... TBD (may need to optimise for this parameter)
            # 'K_adp': 0.05, # mM, Vicini 2000.... TBD (may need to optimise for this parameter)
            'nh': 2.57, # unitless, V/Icini 2000, .... TBD (may need to optimise for this parameter)
            # 'nh': 3, # unitless, Varied

            # Values from Barclay and Weber 2004
            # 'F_0': 0, # N, 
            # 'l_0': 11e-3, # m, 
            # 'mass': 4.1e-3, # g, 
            # Values from Barclay et al. 1995
            'F_0': 0, # N, 
            'l_0':  9.5e-3, # m,    
            'mass': 1.99e-3, # g, 
           
            # 'r_rec': 1 / 0.8 * 60e3, # J / mol, Assumes mitochondrial efficiency based on average mouse sol and edl
            'r_rec': 50000, # F0 l0 / mol, Optimised value
            
            # Heat data used for optimisation 
            'heat_exp_rec': np.array((5.339988e-03, 5.726164e-02, 7.235249e-02, 7.009909e-02)), # J/F0l0, Slow, recovery heat 
            # 'heat_exp_rec': np.array((5.339988e-03, 5.726164e-02, 7.235249e-02, 7.009909e-02)), # J/F0l0, Slow, recovery heat 

            # Barclay and Weber 2004 experimental setup parameters 
            'velo_short': 1.3, # l0/s, Barclay and Weber 2004
            'freq': 250, # Hz, Frequency of stimulation 
            'max_dl': 0.1, # mm, Maximum length change

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
            # For Phillips Simulation 
            'atp_peak': 0.25,# 0.213, # mM/s Peak atp rate calculated based on initial heat rate and enthalpy of ATP from Phillips et al. 1993

            'gamma': 2, # Scaling factor for metabolic rates at rest

            # May need to tune these parameters...
            'K_adp': 0.058, # mM, Vicini 2000.... TBD (may need to optimise for this parameter)
            'nh': 2.57, # unitless, VIcini 2000, .... TBD (may need to optimise for this parameter)
            # 'nh': 1.180, # unitless, Optimised to B1995 dataset 
            # 'nh': 1, # unitless, Varied

            'F_0': 0, # N, 
            'l_0': 8.9e-3, # m,
            'mass': 3.9e-3, # g, 
            
            # 'r_rec': 1 / 0.8 * 60e3, # J / mol, Assumes mitochondrial efficiency based on average mouse sol and edl
            'r_rec': 66536.221, # J / mol, Optimised value
            

            # Heat data used for optimisation 
            'heat_exp_rec': np.array((2.518854e-02, 6.404227e-02, 6.903081e-02, 7.313249e-02)), # J/F0l0, Fast, recovery heat 

            # Barclay and Weber 2004 experimental setup parameters 
            'velo_short': 2.8, # l0/s, Barclay and Weber 2004
            'freq': 100, # Hz, Frequency of stimulation 
            'max_dl': 0.2, # mm, Maximum length change


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

    
}

# Calculate the maximum isometric forces
for muscle_ in ('SOL', 'EDL'):
    # Maximum isometric force (Assuming fixed max_iso_stress for now)    
    params[muscle_]['F_0'] = params[muscle_]['mass'] / params['rho0'] / params[muscle_]['l_0'] * params['max_iso_stress']
    print(f'{muscle_}: Maximum isometric force: {params[muscle_]["F_0"]}')

# Define the initial energy consumption 
def E_init(t): 
    
    trampend = 1
    t_start_cycle = params['t_cycle_start']
    t_end_cycle = params['t_cycle_end']

    # Normalize time
    t_cycle_length = 5 # s, Length of the cycle
    t_cycle = t%t_cycle_length

    
    if params['muscle'] == 'SOL': 
        # Use variable ATP usage 
        def f(x, a, b, c, d): 
            ''' 
            Function as defined in computeParametersBarclay1995.py 
            *** Ensure its the same if any adjustments are made (e.g. fibre-type) ***
            '''
            # return a * np.exp(b * x - c) + d
            return a * b ** x - c
        popt = np.array((0.30521532,  0.65633996, -0.54965355 ,  1))
        # popt = np.array((26.31884038, -0.42107639,  4.45702316,  0.54965355))
        tstimend = 0.8 # s, Length of stimulation (B1995)
    elif params['muscle'] == 'EDL':
        # Use variable ATP usage 
        def f(x, a, b, c, d): 
            ''' 
            Function as defined in computeParametersBarclay1995.py 
            *** Ensure its the same if any adjustments are made (e.g. fibre-type) ***
            '''
            return a * b ** x - c
        # popt = np.array((0.3194931, 0.63699497, -2.01981868, 1.)) 
        popt = np.array((0.3194931,   0.63699497, -2.0198186, 1.)) 
        
        tstimend = 0.2 # s, Length of stimulation (B1995)

    cycle_count = np.floor(t/t_cycle_length) + 1
    # Compute the atp usage based on the cycle number
    atp_peak = f(cycle_count, *popt) # umol/s/(g wet wt) [computed using computeParametersBarclay1995.py]

    return   10**-6 * params['Gatp'] * (atp_peak * (t_cycle < tstimend)) * (t > t_start_cycle) * (t <= t_end_cycle) # J/g/s
          
# Initialise the intial energy 
t_vec = np.linspace(params['t_start'], params['t_end'], 100 * params['t_end'])
t_span = (t_vec[0], t_vec[-1]) 
c_atp_0 = params[params['muscle']]['c_atp_0']
E_tot = E_init(t_vec) # Units of W / g

# Compute the scaled initial energy 
e_init_scale = (params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] / params[params['muscle']]['mass'])**-1
E_tot_scaled = E_tot * e_init_scale # Units of F0l0 / s

#########
# Define the code for the optimisation 
muscle = params['muscle']


# Define the recovery heat from the experimental data 
cycle = np.array((1,5,15,30)) # Cycle numbers
t_exp = params['t_cycle_start'] +  cycle * 5 # s, Times for teh experimental values 
# heat_exp_rec = np.array((5.339988e-03, 5.726164e-02, 7.235249e-02, 7.009909e-02)) # J/F0l0, Slow, recovery heat 
# heat_exp_rec = np.array((2.518854e-02, 6.404227e-02, 6.903081e-02, 7.313249e-02)) # J/F0l0, Fast, recovery heat 
from scipy.interpolate import PchipInterpolator
def rec_heat_exp(t, heat_exp_rec): 
    cspline_exp = PchipInterpolator(t_exp, heat_exp_rec)
    return cspline_exp(t) * (t >= params['t_cycle_start'] + 5) * (t < params['t_cycle_end'])
# cspline_exp = PchipInterpolator(t_exp, heat_exp_rec)
# fig, ax = plt.subplots(layout = 'constrained')
# ax.plot(np.linspace(0,50*5, 100), rec_heat_exp(np.linspace(0,50*5, 100), params['SOL']['heat_exp_rec'])) 
# ax.plot(t_exp, params['SOL']['heat_exp_rec'], '.') 
# plt.show()

# Optimisation function
def f_opt(x):
    ####
    # Define the bioenergetics model 
    model = Bioenergetics(params)
    model.r_rec = x[0] * 1e6
    model.nh = x[1]
    model.K_adp = x[2]
    model.V_max_oxphos = x[3]
    # print(f'r_rec = {model.r_rec}, nh = {model.nh}, K_adp = {model.K_adp}, V_max_oxphos = {model.V_max_oxphos}')

    #####
    # Solve the ode 
    # t_span = (params['t_start'],params['t_end']) 
    sol = model.solveBioenergetics(t_span, c_atp_0, t_vec, E_tot) # Note E_tot here in W/g
    # PCr, ADP, activation = sol.T  # Transpose to get individual variables
    c_atp, c_pcr = sol.y  # Transpose to get individual variables

    # Compute the recovery 
    recovery_heat_model_ = model.computeRecoveryEnergetics(sol.t, sol.y[0,]) # F0 l0 / g / s 

    # Scale the recovery heat to units of F0 l0 / s
    scale_factor = params[muscle]['mass'] 
    recovery_heat_model = recovery_heat_model_ # Units of F0 l0/ s

    # compute the recovery heat from the experiment 
    recovery_heat_exp = rec_heat_exp(sol.t, params[muscle]['heat_exp_rec']) # Units of F0 l0/s

    # only fit over the times from t_cycle_start to t_cycle_end 
    recovery_heat_model_crop = recovery_heat_model[(sol.t > params['t_cycle_start'] + 5) * (sol.t < params['t_cycle_end'])]
    recovery_heat_exp_crop = recovery_heat_exp[(sol.t > params['t_cycle_start']+ 5) * (sol.t < params['t_cycle_end'])]
    
    ################################
    # Opt 1: just minimise over initial time period 
    # error = np.linalg.norm(recovery_heat_model_crop - recovery_heat_exp_crop) / np.linalg.norm(recovery_heat_exp_crop)

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

    # ################################
    # # Opt 3: TODO add in a condition to minimise the derivative
    error_init_per = np.linalg.norm(recovery_heat_model_crop - recovery_heat_exp_crop) / np.linalg.norm(recovery_heat_exp_crop)
    E_init_final = np.trapz(E_tot_scaled, t_vec) # Get final initial energy 
    E_rec_final = np.trapz(recovery_heat_model, t_vec) # Get final recovery energy 
    error_ratio_final = np.linalg.norm(E_init_final /  E_rec_final - 1) / np.linalg.norm(E_init_final)
    E_rec_deriv = np.diff(recovery_heat_model) / np.diff(t_vec)
    error_deriv_final = np.abs(E_rec_deriv[-1])
    w1 = x[4] * 10 # Weighting (not sure if needed)
    w2 = x[5] * 10# Weighting for derivative
    error = error_init_per + w1 * error_ratio_final + w2 * error_deriv_final

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
    print(f'iter: {iter}, r_rec = {xk[0]}, nh = {xk[1]}, K_adp = {xk[2]}, V_max_oxphos = {xk[3]}, w0 = {xk[4]}, w1 = {xk[5]}')
    # print(f'iter: {iter}, r_rec = {xk[0]}')
    return 0 

# Define initial conditions and bounds for parameters 
w0_0, w1_0 = 1,1
x0 = (params[muscle]['r_rec'] / 1e6, params[muscle]['nh'], params[muscle]['K_adp'], params[muscle]['V_max_oxphos'], w0_0, w1_0)
bounds_ = ((0.01,4), (0.5,5), (0.001,0.4), (0.001,5), (0.001,5), (0.1,100))

# Perform the optimisation
opt_res = minimize(f_opt, x0, callback= callback_fun, bounds = bounds_,  options = {'maxiter': 500, 'disp': True}, method='Nelder-Mead')
# Print out the optimal solution 
x = opt_res.x 
print(f'r_rec = {x[0]}, nh = {x[1]}, K_adp = {x[2]}, V_max_oxphos = {x[3]}')
# update the parameter values , note multiplication factors
params[muscle]['r_rec'], params[muscle]['nh'], params[muscle]['K_adp'], params[muscle]['V_max_oxphos'] = x[0] * 1e6, x[1], x[2], x[3] 

# # Define initial conditions and bounds for parameters 
# x0 = (params[muscle]['r_rec'] * 100,)
# bounds_ = ((0.001, 100000000), )
# # Perform the optimisation
# opt_res = minimize(f_opt, x0, callback= callback_fun, bounds = bounds_,  options = {'maxiter': 1000, 'disp': True}, method='Nelder-Mead')
# # Print out the optimal solution 
# x = opt_res.x 
# print(f'r_rec = {x[0]}')
# # update the parameter values 
# params[muscle]['r_rec'] = x[0]


# Rerun the model with the optimal values 
bioenergetic_model = Bioenergetics(params) 
# Solve the model
sol = bioenergetic_model.solveBioenergetics(t_span, c_atp_0, t_vec, E_tot)
# Compute the energetic rates 
# scale =  params[params['muscle']]['mass'] / params[params['muscle']]['F_0'] / params[params['muscle']]['l_0'] 
# scale =  params[params['muscle']]['mass'] 
scale = 1
q_r = bioenergetic_model.computeRecoveryEnergetics(sol.t, sol.y[0,]) * scale # In units of W/F0l0

# Compare model and experimental recovery heat rates
recovery_heat_exp = rec_heat_exp(sol.t, params[muscle]['heat_exp_rec'])
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(sol.t, q_r, label='Model recovery heat rate')
ax.plot(sol.t, recovery_heat_exp, label='Experimental recovery heat rate (interp.)', linestyle='--')
ax.plot(t_exp, params[muscle]['heat_exp_rec'], 'o', label='Experimental data points')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Recovery heat rate ($F_0 l_0 s^{-1}$)')
ax.legend()
ax.grid(True)

# Plot with units in F0l0/s
# Plot the rates 
fig, ax = plt.subplots(layout = 'constrained')
# e_init_scale = (params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] / params[params['muscle']]['mass'])
# E_tot = E_tot * e_init_scale
# energy_unit_scaler = params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] / params[params['muscle']]['mass'] * 1e3 # convert from W/F0l0 to mW/g 
energy_unit_scaler = 1 
ax.plot(t_vec, (E_tot_scaled + q_r) * energy_unit_scaler, label = '$\dot q_r + \dot e_{init}$') 

ax.plot(t_vec, E_tot_scaled * energy_unit_scaler , label = '$\dot e_{init}$') 
ax.plot(t_vec, q_r * energy_unit_scaler, label = '$\dot q_r$') 
ax.legend()
ax.set_xlabel('Time (s)')
# ax.set_ylabel('Energy rate ($mW g^{-1}$)')
ax.set_ylabel('Energy rate ($F_0 l_0 s^{-1}$)')
# Plot the total energy over the cycle
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, cumtrapz(E_tot_scaled + q_r, t_vec, initial = 0) * energy_unit_scaler, label = '$ q_r + e_{init}$') 

ax.plot(t_vec, cumtrapz(E_tot_scaled, t_vec, initial = 0) * energy_unit_scaler, label = '$ e_{init}$') 
ax.plot(t_vec, cumtrapz(q_r, t_vec, initial = 0) * energy_unit_scaler, label = '$ q_r$') 
ax.legend()
ax.set_xlabel('Time (s)')
# ax.set_ylabel('Energy  ($mJ g^{-1}$)')
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


# # Match the ratio of initial to recovery heat rates after cycle 30.
# sol_ratio = 4.170472e-01
# edl_ratio = 1.203944e-01

# # Optimise r_r to the soleus 
# def opt_fun(x): 
#     Q_r_ = model.computeEnergetics(PCr, activation, x)
#     Q_r = np.mean(Q_r_[sol.t > 145])
#     Q_init = np.max(model.ATP(np.linspace(140,150,100)) * 60e3 / 10**6) # Get the initial heat as the maximum heat rate over the last two trials 
#     return np.abs(Q_r / Q_init - edl_ratio)
# from scipy.optimize import minimize_scalar
# opt_sol = minimize_scalar(opt_fun, (0.001,10)) 

# # Update the recovery heat rate
# params['Energetics_model']['r_r'] = opt_sol.x
# print(f'Optimal r_r vaue based on ratio: {opt_sol.x}')


# Q_r = model.computeEnergetics(PCr, activation, params['Energetics_model']['r_r']) # Compute the recovery energetic rate

# # Optimise the model recovery ratio to the dataset from 

# # Plot the comparison between PCr experimental and modelled 
# fig, ax  = plt.subplots(figsize = (6,4), layout = 'constrained') 
# ax.plot(sol.t, PCr, label = 'PCr (model)', color = 'k')
# # ax.plot(t_exp, pcr_exp, label = 'PCr (Phillip et al. 1993)', ls = 'None', marker = '.', color = 'k')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('PCr Concentration [umol/g]')
# plt.legend()
# # plt.savefig('./Figures/B1995PCr.pdf')

# # Plot the energetic rates (experimental and predicted rates) 
# fig, ax = plt.subplots(figsize = (6,4), layout = 'constrained')
# ax.plot(sol.t, Q_r, label = '\dot Q_r (model)', color = 'k')
# ax.plot(sol.t, model.ATP(sol.t) * 60e3 / 10**6, label = '\dot Q_i', color = 'k', ls = 'dashed') 
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Recovery energetic rate (W/g)')
# plt.legend()
# # plt.savefig('./Figures/B1995Energetics.pdf')


# plt.show()

