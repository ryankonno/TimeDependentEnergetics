'''
Code to determine the activation and maintenance heat parameters for the model

This code uses twitch data from Barclay 2012

'''

# Import 
import numpy as np 
from scipy.integrate import solve_ivp, cumtrapz
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt 
font = {'size'   : 14}
plt.rc('font', **font)
import matplotlib.cm as cmap
import pandas as pd
import itertools
import sys 
sys.path.append('./')


# Data files for each muscle type (SOL = soleus/slow, EDL = fast)
# Time varying data from Barclay 2012
data_files = {
    'SOL': {
        'TOT':'Data/B2012_data_twitch_heat_SOL_Total.csv',
        'ACT':'Data/B2012_data_twitch_heat_SOL_Act.csv',
    },
    'EDL': {
        'TOT':'Data/B2012_data_twitch_heat_EDL_Total.csv',
        'ACT':'Data/B2012_data_twitch_heat_EDL_Act.csv',
    }
}

# Mean peak heat output from fast and slow values (digitised from Barclay 2012)
# Data in mJ/g at 30 deg C
data_exp = {
    'SOL': {
        'TOT': 3.74,
        'ACT': 1.75,
    },
    'EDL': {
        'TOT': 6.74,
        'ACT': 3.15,
    }
}




# Define the model parameters
params = {
    # Time parameters for setting up the protocol 
    't_start': 0, # s
    't_end': 2, # s

    # General muscle parameters
    'rho0':  1e6,    # g/m^3, Density of muscle
    # 'max_iso_stress': 2.5e5, # N/m^2, Maximum isometric stress of the muscle

    'muscle': 'SOL', # Specify muscle parameters to be used in simulation

        # Mouse data 
        'SOL': {
            'mass': 4.38e-3, # g, B2012
            'l_0': 13.2e-3, # m, B2012
            # 'max_iso_stress': 28.8e3 * 7.692, # N/m^2, B2012, scaled assuming a twitch-tetanus ratio fo 0.13 (Celichowski 1993 )
            'max_iso_stress': 2.37e5, # N/m^2, B1996

            # # # Activation model parameters 
            # # # "Tau_1": 0.0422, # Assume constant value from MCL (2023)
            # # "Tau_1": 0.038, # BH2012 @ 30deg
            # # # "Tau_1":  0.011 * 0.5, # BH2012 @ 20deg (assuming half release with tem (B2012))
            # # # "Tau_2": 0.125, # Scaling based on MCL (2023)
            # # # "Tau_2": 0.057, #  BH 2012        
            # # # "Tau_2": 0.065, # B2012 20deg
            # # "Tau_2": 0.055, # B2012 30deg   
            # # "Tau_1": 0.022, # BH 2003, Fibre bundle data 
            # # "Tau_2": 0.408 , # BH 2003, Fibre bundle data
            # # Using approximat values from experiment
            # "Tau_1": 0.001, # Based on Ca of second twitch
            # "Tau_2": 0.055, # B2012 30deg 
            
            # # "K": 0.1025,
            # "K": 0.9,
            # "n": 1.99 , # Hill coefficient for act mdoel, Lynch XX
            # # "n": 2.55, # Hill coefficient for act mdoel, Curtin et al., 1998

            # Activation model parameters 
            'Tau_1': 0.038,  # requested
            'Tau_2': 0.055,  # B2012 30deg
            "K": 0.2,
            "n": 1.99, # Hill coefficient for act mdoel

            # Mechanical parameters 
            'dedt_ce_max': 5, 
            'kappa': 0.18,

            # Initial energetics model 
            # 'r_am': 0.6177, # W/F_0/l_0, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cxb': 0.6 * 0.6177/ 4, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cat': 0.4 * 0.6177/ 4, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_sl': 0.234/ 4, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

            # Q10 factors for the heat rates, values from B2012 
            'q10_actheat': 1.75, 
            'q10_cxbheat': 1.4,
            'q10_totheat': 1.5,

        }, 
        'EDL': { 
            'mass': 3.44e-3, # g, B2012
            'l_0': 8.55e-3, # m, B2012
            # 'max_iso_stress': 36.0e3 * 5, # N/m^2, B2012, assuming a twitch to tetanus ration of 0.2 (Celichowski 1993)
            'max_iso_stress': 3.01e5, # N/m^2, B1996
            
            # # Activation model parameters 
            # # "Tau_1": 0.0422, # Very little change between fibre type - assume constant (BH, 2003)
            # "Tau_1": 0.011, # bh 2012  @ 30deg
            # # "Tau_1": 0.011, # bh 2012  @ 20deg (assuming half release with tem (B2012))
            # # "Tau_2": 0.011, # BH 2012
            # # "Tau_2": 0.032, # B2012 20deg
            # "Tau_2": 0.013, # B2012 30deg
            # "Tau_1": 0.008, # BH 2003, Fibre bundle data 


            # "Tau_1": 0.0006, # obatined ca ratio second twitch
            # "Tau_2": 0.013, # B2012 30deg

            # "K": 0.9,
            # # "n": 3, # Hill coefficient for activation model
            # # "n": 2.89, # Hill coefficient for act mdoel, Curtin et al., 1998
            # "n": 2.89, # Hill coefficient for act mdoel, Curtin et al., 1998


            #Original values
            'Tau_1': 0.011,  # requested
            'Tau_2': 0.011,  # BH 2003, fibre bundle data
            "K": 0.35,
            "n": 2.89, # Hill coefficient for activation model

            # Mechanical parameters 
            'dedt_ce_max': 10, 
            'kappa': 0.25,

            # Energetics model 
            'r_cxb': 0.6 * 2.792, # F0l0/s, Maximum heat rate of isometric contraction (fast-type fibre)
            'r_cat': 0.4 * 2.792, # F0l0/s, Maximum heat rate of isometric contraction (fast-type fibre)
            'r_sl': 0.697, # F0l0/s, Maximum shortening heat rate (fast-type fibre)

            # Q10 factors for the heat rates, values from B2012 
            'q10_actheat': 1.957, 
            'q10_cxbheat': 1.85,
            'q10_totheat': 1.89,

        },

        # Mechanical parameters (may not be used)
        'k_see': 0, # Unused
    
}

#####
# Load the models
from Models.MUActivationModel import ActivationModel
from Models.MechanicsModelSimple import MechModel 
from Models.MUEnergeticsModelSimple_SplitVars import EnergeticsModel


# Store data from each muscle so both can be plotted side by side.
activation_plot_data = {}
heat_plot_data = {}


########
# Loop over both muscle types: SOL (soleus/slow) and EDL (fast)
for muscle_type in ['SOL', 'EDL']:
    params['muscle'] = muscle_type
    muscle = muscle_type

    print(f'\n=== Processing {muscle_type} ({"slow/soleus" if muscle_type == "SOL" else "fast/EDL"}) ===')

    ########
    # Compute necessary constants 
    params[muscle]['F_0'] = params[muscle]['mass'] / params['rho0'] / params[muscle]['l_0'] * params[muscle]['max_iso_stress']

    # Print constants used in normalisation and the normalisation factor to F_0 l_0 
    print(f'mass = {params[muscle]["mass"]}')
    print(f'F_0 = {params[muscle]["F_0"]}')
    print(f'l_0 = {params[muscle]["l_0"]}')
    print(f'Scale factor = {params[muscle]["mass"] / params[muscle]["F_0"] / params[muscle]["l_0"] * 1e-3}')
    

    ########
    # Simulate the Ca dynamics 
    t_vec = np.linspace(params['t_start'], params['t_end'], int(50000 * params['t_end']))

    # Define the models
    act_model = ActivationModel(params[params['muscle']], t_vec, True)
    mech_model = MechModel(1, params[muscle]['dedt_ce_max'], params[muscle]['kappa'], params['k_see'])
    energy_model = EnergeticsModel()
    
    # Solve the activation model with the computed stim values
    idx_stims = np.array((0,))
    stim_vec, ca_vec, catn_vec = act_model.runExcAct(idx_stims, w_0 = 0.004) # Implemented with a 1s stimulus
    # stim_vec, ca_vec, catn_vec = act_model.runExcAct(idx_stims)

    # Peak Ca released after one twitch
    print(f'Peak Ca: {np.max(ca_vec):.6e} (normalised units)')

    activation_plot_data[muscle] = {
        't_vec': t_vec.copy(),
        'ca_vec': ca_vec.copy(),
        'catn_vec': catn_vec.copy(),
    }


    # #####
    # # Get the experimental heat values 
    # # Load the experimental data for this muscle
    # data_exp_act = pd.read_csv(data_files[muscle_type]['ACT'])
    # data_exp_tot = pd.read_csv(data_files[muscle_type]['TOT'])

    # # Interpolate the the experimental heat 
    # data_exp_act_interp = np.interp(t_vec, data_exp_act['Time'], data_exp_act['Heat'])
    # data_exp_tot_interp = np.interp(t_vec, data_exp_tot['Time'], data_exp_tot['Heat'])
    # data_exp_cxb_interp = data_exp_tot_interp - data_exp_act_interp

    # # Account for q10 scaling
    # data_exp_act_interp *= params[muscle]['q10_actheat']
    # data_exp_tot_interp *= params[muscle]['q10_totheat']
    # data_exp_cxb_interp = data_exp_tot_interp - data_exp_act_interp
    
    # # scale the heat values to units of F_0 l_0 from mJ/g
    # data_exp_act_interp *= params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0'] * 1e-3
    # data_exp_tot_interp *= params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0'] * 1e-3
    # data_exp_cxb_interp *= params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0'] * 1e-3

    


    # # Plot the raw data
    # fig,ax = plt.subplots(layout = 'constrained', figsize= (7,5))
    # ax.plot(t_vec, data_exp_tot_interp, label = 'tot')
    # ax.plot(t_vec, data_exp_act_interp, label = 'cat')
    # ax.plot(t_vec, data_exp_cxb_interp, label = 'cxb')
    # ax.legend()
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Heat $mJ g^{-1}$')

    # Use final heat values  (after 2s) 
    data_exp_act = data_exp[muscle]['ACT']
    data_exp_tot = data_exp[muscle]['TOT']
    data_exp_cxb = data_exp_tot - data_exp_act
    
    # scale the heat values to units of F_0 l_0 from mJ/g
    data_exp_act *= params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0'] * 1e-3
    data_exp_tot *= params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0'] * 1e-3
    data_exp_cxb *= params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0'] * 1e-3

    


    # Define the optimisation function for cat
    def fun_opt_cat(x):

        # Update values for parameters
        params[muscle]['r_cat'] = x

        # set up the mechanical conditions
        dedt_ce = np.zeros_like(t_vec)
        e_ce = np.zeros_like(dedt_ce) # NOTE: We choose no strain since we are at steady state and shortening over plateau 

        # Get the force
        force = catn_vec * mech_model.F_va(dedt_ce) * mech_model.F_la(e_ce)
            
        # Compute the energy rates 
        q_a, q_m, q_sl, w = energy_model.actEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force, mech_model)

        # Compute the total activation heat 
        q_a_int = cumtrapz(q_a, t_vec, initial=0)


        # Compute the error between the computed heat and the experimental activation heat 
        # error = np.linalg.norm(data_exp_act_interp - q_a_int)**2
        error = (data_exp_act - q_a_int[-1])**2

        return error 

    # Define the optimisation function for cxb
    def fun_opt_cxb(x):

        # Update values for parameters
        params[muscle]['r_cxb'] = x

        # set up the mechanical conditions
        dedt_ce = np.zeros_like(t_vec)
        e_ce = np.zeros_like(dedt_ce) # NOTE: We choose no strain since we are at steady state and shortening over plateau 

        # Get the force
        force = catn_vec * mech_model.F_va(dedt_ce) * mech_model.F_la(e_ce)
            
        # Compute the energy rates 
        q_a, q_m, q_sl, w = energy_model.actEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force, mech_model)

        # Compute the total activation heat 
        q_m_int = cumtrapz(q_m, t_vec, initial=0)


        # Compute the error between the computed heat and the experimental activation heat 
        # error = np.linalg.norm(data_exp_cxb_interp - q_m_int)**2
        error = (data_exp_cxb - q_m_int[-1])**2

        return error 
    

    # Perform the optimisation for the activation coeffitient 
    print(f'Running activation optimisation')
    opt_res_act = minimize_scalar(fun_opt_cat)
    # print(opt_res_act)
    print(f'Running cross-bridge optimisation')
    opt_res_cxb = minimize_scalar(fun_opt_cxb)
    # print(opt_res_cxb)

    params[muscle]['r_cat'] = opt_res_act.x
    params[muscle]['r_cxb'] = opt_res_cxb.x
    # params[muscle]['r_sl'] = opt_res.x[2]

    print('Optimised parameters:')
    print(f'r_cxb = {params[muscle]["r_cxb"]}, r_cat = {params[muscle]["r_cat"]}')

    # recompute the values 
    # set up the mechanical conditions
    dedt_ce = np.zeros_like(t_vec)
    e_ce = np.zeros_like(dedt_ce) 
    # Get the force
    force = catn_vec * mech_model.F_va(dedt_ce) * mech_model.F_la(e_ce)
    # Compute the energy rates 
    q_a, q_m, q_sl, w = energy_model.actEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force, mech_model)

    # Compute the total activation heat 
    q_m_int = cumtrapz(q_m, t_vec, initial=0)
    q_a_int = cumtrapz(q_a, t_vec, initial=0)

    heat_plot_data[muscle] = {
        't_vec': t_vec.copy(),
        'q_a_int': q_a_int.copy(),
        'q_m_int': q_m_int.copy(),
        'data_exp_act': data_exp_act,
        'data_exp_cxb': data_exp_cxb,
    }


# Plot activation side by side for both muscles.
fig_act, axes_act = plt.subplots(1, 2, layout='constrained', figsize=(12, 4.5), sharey=True)
fig_act.suptitle('Activation Dynamics')
for ax, muscle in zip(axes_act, ['SOL', 'EDL']):
    muscle_data = activation_plot_data[muscle]
    ax.plot(muscle_data['t_vec'], muscle_data['ca_vec'], label='$Ca^{2+}$')
    ax.plot(muscle_data['t_vec'], muscle_data['catn_vec'], label='$CaTn$')
    ax.set_title(muscle)
    ax.set_xlabel('Time ($s$)')
axes_act[0].set_ylabel('Normalised concentration')
axes_act[0].legend()


# Plot activation and cross-bridge heat side by side for both muscles.
fig_heat, axes_heat = plt.subplots(1, 2, layout='constrained', figsize=(12, 4.5), sharey=True)
fig_heat.suptitle('Activation and Cross-Bridge Heat')
for ax, muscle in zip(axes_heat, ['SOL', 'EDL']):
    muscle_data = heat_plot_data[muscle]
    ax.plot(muscle_data['t_vec'], muscle_data['q_a_int'], label='Act-Model', color='k')
    ax.plot(muscle_data['t_vec'], muscle_data['q_m_int'], label='Cxb-Model', color='r')
    ax.axhline(muscle_data['data_exp_act'], color='k', ls='--', lw=1.2, label='Act-Experiment')
    ax.axhline(muscle_data['data_exp_cxb'], color='r', ls='--', lw=1.2, label='Cxb-Experiment')
    ax.set_title(muscle)
    ax.set_xlabel('Time ($s$)')
axes_heat[0].set_ylabel('Heat ($F_0 l_0$)')
axes_heat[0].legend()


plt.show()


