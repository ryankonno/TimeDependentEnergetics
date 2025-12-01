'''
This code implements the model as designed in Kushmerick 1999, and adapted in Vicini 2000

This version of the code has been adapted such that the initial resting rate of the model will altered based of Vmax. This allows for a fixed c_pcr_0 while othere parameters are varied... I do not believe it is expected that if rate constants vary then there shoulld be an alteratation in steady-state metabolism

This code fits the recovery heat parameter + ATP Pcr mdoel parameters to the Barclay 1995 dataset

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
import itertools
import sys 
sys.path.append('./')
# Parameters
params = {
    # Time parameters for setting up the protocol 
    't_start': 0, # s
    't_end': 60, # s
    'cycle_length': 0.25, # s, Defines the length of the cycle (used to set frequency of the contractions)
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
            # 'Pi_0': 6, # mM,  Kushmerick et al. 1992 
            'V_max_oxphos':  1.88, # mM/s, Vicini 2000... TBD
            # For Phillips Simulation 
            'atp_peak': 0.25, # 0.213, # mM/s Peak atp rate calculated based on initial heat rate and enthalpy of ATP from Phillips et al. 1993

            # May need to tune these parameters...
            'K_adp': 0.058, # mM, Vicini 2000.... TBD (may need to optimise for this parameter)
            # 'nh': 2.57, # unitless, VIcini 2000, .... TBD (may need to optimise for this parameter)
            'nh': 0.6456, # unitless, Varied

            # Values from Barclay and Weber 2004
            'F_0': 0, # N, 
            'l_0': 11e-3, # m, 
            'mass': 4.1e-3, # g, 
           
            # 'r_rec': 1 / 0.8 * 60e3, # J / mol, Assumes mitochondrial efficiency based on average mouse sol and edl
            'r_rec': 149835.87, # J / mol, Optimised value
            
            # Heat data used for optimisation 
            'heat_exp_rec': np.array((5.339988e-03, 5.726164e-02, 7.235249e-02, 7.009909e-02)), # J/F0l0, Slow, recovery heat 

            # Barclay and Weber 2004 experimental setup parameters 
            'velo_short': 1.3, # l0/s, Barclay and Weber 2004
            'freq': 250, # Hz, Frequency of stimulation 
            'max_dl': 0.1, # mm, Maximum length change

            # Activation model parameters 
            "Tau_1": 0.3, # Assume constant value from MCL (2023)
            # "Tau_2": 0.256, # Scaling based on MCL (2023)
            "Tau_2": 0.02,
            "K": 0.1025,
            "n": 4, # Hill coefficient for act mdoel

            
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

            # May need to tune these parameters...
            'K_adp': 0.058, # mM, Vicini 2000.... TBD (may need to optimise for this parameter)
            # 'nh': 2.57, # unitless, VIcini 2000, .... TBD (may need to optimise for this parameter)
            'nh': 1.180, # unitless, Varied

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
            "n": 3, # Hill coefficient for activation model

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

'''
Compute necessary parameters from data 
'''

for muscle in ('SOL', 'EDL'):
    # Maximum isometric force (Assuming fixed max_iso_stress for now)    
    params[muscle]['F_0'] = params[muscle]['mass'] / params['rho0'] / params[muscle]['l_0'] * params['max_iso_stress']
    print(f'{muscle}: Maximum isometric stress: {params[muscle]["F_0"]}')

'''
Setup the simulation 
'''
def f_stim_length(t, params): 
    # Function to compute the length changes in the muscle 
    # returns both simulation times and lengths 

    cycle_length = params['cycle_length'] # s, Set at 0.3s for now (as in figure 1)
    t_cycle = t % cycle_length # Get the time with respect to the cycle
    N_cycles = params['N_cycles'] # Number of cycles to simulate

    # time parameters (with respect to cycle time )
    t_stim_start = 0 
    t_stim_end = 0.2 * cycle_length
    t_short_start = 0.12 * cycle_length
    t_length_start = 0.4 * cycle_length
    t_length_end =  cycle_length # Assume return to initial lenght by the end of the cycle

    # OPT: Linear implementation
    # Extract necessary parameter values & compute velocities
    l_0 = params[params['muscle']]['l_0']
    dl_max = params[params['muscle']]['max_dl']  # Maximum length change
    v_short = dl_max * l_0 / (t_length_start  - t_short_start) # shortening velocity
    v_length = dl_max * l_0 / (t_length_end - t_length_start) # lengthening velocity
    print(f'v_short = {v_short / l_0} l0/s')

    # Change in length (mm)
    dl = ((t_cycle > t_short_start) * (t_cycle < t_length_start) * (-v_short * (t_cycle - t_short_start))\
            + (t_cycle > t_length_start) * (t_cycle < t_length_end) * (-v_short * (t_length_start - t_short_start) + v_length * (t_cycle - t_length_start)))\
            * (t < cycle_length * N_cycles)

    # OPT: Sinusoidal implementation
    # # Extract necessary parameter values & compute velocities
    # l_0 = params[params['muscle']]['l_0']
    # dl_max = params[params['muscle']]['max_dl']  # Maximum length change
    # print(f'v_short = {dl_max * l_0 / (t_length_start - t_short_start) / l_0} l0/s')

    # # Change in length using smooth sinusoidal motion
    # # Shortening phase: sinusoidal from 0 to -dl_max
    # t_short_duration = t_length_start - t_short_start
    # dl_short = np.where(
    #     (t_cycle >= t_short_start) & (t_cycle < t_length_start),
    #     -dl_max * (1 - np.cos(np.pi * (t_cycle - t_short_start) / t_short_duration)) / 2,
    #     0
    # )

    # # Lengthening phase: sinusoidal from -dl_max back to 0
    # t_length_duration = t_length_end - t_length_start
    # dl_length = np.where(
    #     (t_cycle >= t_length_start) & (t_cycle <= t_length_end),
    #     -dl_max + dl_max * (1 - np.cos(np.pi * (t_cycle - t_length_start) / t_length_duration)) / 2,
    #     0
    # )

    # # Combine both phases
    # dl = dl_short + dl_length


    # Toggle whether in stimulation or not (does not define frequency of stim here)
    stim = ((t_cycle >= t_stim_start) * (t_cycle <= t_stim_end) ) * (t < cycle_length * N_cycles)

    # Compute the stimulation times 
    # stim_times: vector (same shape as t) with 1 where a stimulus (spike) occurs, 0 otherwise
    stim_times = np.zeros_like(t, dtype=int)

    # determine stimulation frequency from params for the active muscle (fallback to 0)
    freq = params[params['muscle']]['freq']

    if freq > 0:
        period = 1.0 / freq

        # Get times when there is a stimulus 
        t_stim_period = t[stim] 
        # print(t_stim_period)

        # Get the firing times (accumulate into a Python list then convert)
        t_fire_vec = []
        if t_stim_period.size > 0:
            t_fire_prev = t_stim_period[0]
            t_fire_vec.append(t_fire_prev)
            # iterate remaining times and add a spike when period elapsed
            for t_val in t_stim_period[1:]:
                if t_val >= t_fire_prev + period - 1e-12:
                    t_fire_vec.append(t_val)
                    t_fire_prev = t_val
        t_fire_vec = np.array(t_fire_vec)

        # Map each spike time to the nearest index in t and mark stim_times
        stim_times = np.zeros_like(t, dtype=int)
        if t_fire_vec.size > 0:
            t_arr = np.asarray(t)
            for st in t_fire_vec:
                idx = int(np.argmin(np.abs(t_arr - st)))
                stim_times[idx] = 1

        # optional debug prints
        # print('spike times:', t_fire_vec)
        # print('stim_times vector sum:', stim_times.sum())
            
    return stim, stim_times, dl

# Plot to verify conditions 
t_vec = np.linspace(params['t_start'], params['t_end'], int(1000 * params['t_end'])) 
stim_vec, stim_times_vec,  dl_vec = f_stim_length(t_vec, params)
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, dl_vec, color = 'black')
ax2 = ax.twinx()
ax2.plot(t_vec, stim_vec, color = 'red')
ax2.plot(t_vec, stim_times_vec, marker = '*',  color = 'red')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Length change (m)')
ax2.set_ylabel('Simulus', color = 'red')
ax2.set_ylim((0,3))
plt.show()

''' 
Simulate Ca2+ and mechanics 
'''

# Ca dynamics
from Models.MUActivationModel import ActivationModel
act_model = ActivationModel(params[params['muscle']], t_vec, True)
idx_stims = np.nonzero(stim_times_vec)[0]
stim_vec, ca_vec, catn_vec = act_model.runExcAct(idx_stims)
# Plot the results 
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, ca_vec, label = 'Free Ca') 
ax.plot(t_vec, catn_vec, label = 'CaTn')
# ax.plot(stim_vec)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Normalised concentration')
plt.show()


# Mechanics 
from Models.MechanicsModelSimple import MechModel 
muscle = params['muscle']
mech_model = MechModel(params[muscle]['l_0'], params[muscle]['dedt_ce_max'], params[muscle]['kappa'],params['k_see'])
# Compute the strain and strain rates in the muscle 
e_ce = dl_vec / params[muscle]['l_0'] + params[params['muscle']]['max_dl'] / 2 # Get the strain adjusted so length change is over plateau
dedt_ce = np.diff(e_ce, prepend = 0) / np.diff(t_vec, prepend = 1)
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, e_ce) 
ax.plot(t_vec, dedt_ce) 
ax.set_xlabel('Time (s)')
ax.set_ylabel('Strain/strain raet')
plt.show()

# Plot the force
# Compute the force directly  
force_direct =  mech_model.computeForce(catn_vec, e_ce + 1, dedt_ce)
# Compute the force accounting for damping 
force_damped = mech_model.solveModel(t_vec, catn_vec, e_ce + 1, dedt_ce, params[muscle]['l_0'], \
                                     params[muscle]['mass'], damp_coeff= 10000, stiff_coeff= 100,\
                                          l_see_rat=0.1)
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, force_direct, label = 'direct') 
ax.plot(t_vec, force_damped, label = 'damped')
ax.plot(t_vec, catn_vec, label = 'activation')
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Force (normalised)')
plt.show()


# Compute the initial energetics 
from Models.MUEnergeticsModelSimple import EnergeticsModel
energy_model = EnergeticsModel()
q_a, q_m, q_sl, w = energy_model.actEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force_direct, mech_model)
E_tot = q_a + q_m + q_sl + w  # Total energy 
# Plot the rates 
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, q_a, label = '$\dot q_a$') 
ax.plot(t_vec, q_m, label = '$\dot q_m$') 
ax.plot(t_vec, q_sl, label = '$\dot q_{sl}$') 
ax.plot(t_vec, w, label = '$\dot w$')
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Energy rate ($W \, (F_0 l_0)^{-1}$)')
# Plot the total energy over the cycle
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, cumtrapz(q_a, t_vec, initial = 0), label = '$ q_a$') 
ax.plot(t_vec, cumtrapz(q_m, t_vec, initial = 0), label = '$ q_m$') 
ax.plot(t_vec, cumtrapz(q_sl, t_vec, initial = 0), label = '$ q_{sl}$') 
ax.plot(t_vec, cumtrapz(w, t_vec, initial = 0), label = '$ w$')
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Energy  ($J \, (F_0 l_0)^{-1}$)')

# Convert units to input for bioenergetics model 
E_initial_converted = E_tot * params[muscle]['F_0'] * params[muscle]['l_0'] / params[muscle]['mass'] # W/g

# Run bioenergetics
from Models.BioenergeticsSimple import Bioenergetics
bioenergetic_model = Bioenergetics(params) 
t_span = (t_vec[0], t_vec[-1]) 
c_atp_0 = params[muscle]['c_atp_0']
# Solve the model
sol = bioenergetic_model.solveBioenergetics(t_span, c_atp_0, t_vec, E_initial_converted)
# Compute the energetic rates 
scale =  params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0'] 
q_r = bioenergetic_model.computeRecoveryEnergetics(sol.y[0,]) * scale # In units of W/F0l0
# Plot the rates 
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, E_tot, label = '$\dot e_{init}$') 
ax.plot(t_vec, q_r, label = '$\dot q_r$') 
# ax.plot(t_vec, (E_tot + q_r), label = '$\dot q_r + \dot e_{init}$') 
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Energy rate ($W \, (F_0 l_0)^{-1}$)')
# Plot the total energy over the cycle
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, cumtrapz(E_tot, t_vec, initial = 0), label = '$ e_{init}$') 
ax.plot(t_vec, cumtrapz(q_r, t_vec, initial = 0), label = '$ q_r$') 
ax.plot(t_vec, cumtrapz(E_tot + q_r, t_vec, initial = 0), label = '$ q_r + e_{init}$') 

ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Energy  ($J \, (F_0 l_0)^{-1}$)')

# Plot with units in mW/g 
# Plot the rates 
fig, ax = plt.subplots(layout = 'constrained')
energy_unit_scaler = params[muscle]['F_0'] * params[muscle]['l_0'] / params[muscle]['mass'] * 1e3 # convert from W/F0l0 to mW/g 
ax.plot(t_vec, E_tot * energy_unit_scaler, label = '$\dot e_{init}$') 
ax.plot(t_vec, q_r * energy_unit_scaler, label = '$\dot q_r$') 
ax.plot(t_vec, (E_tot + q_r) * energy_unit_scaler, label = '$\dot q_r + \dot e_{init}$') 
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Energy rate ($mW g^{-1}$)')
# Plot the total energy over the cycle
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, cumtrapz(E_tot, t_vec, initial = 0) * energy_unit_scaler, label = '$ e_{init}$') 
ax.plot(t_vec, cumtrapz(q_r, t_vec, initial = 0) * energy_unit_scaler, label = '$ q_r$') 
ax.plot(t_vec, cumtrapz(E_tot + q_r, t_vec, initial = 0) * energy_unit_scaler, label = '$ q_r + e_{init}$') 
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Energy  ($mJ g^{-1}$)')


# Plot the fluxes
fig_flux, axs_flux = plt.subplots(1,3, layout = 'constrained', figsize = (14,5)) 
ax = axs_flux[0] 
ax.plot(sol.t, bioenergetic_model.phi_atp(sol.t, sol.y[0,]))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Flux ATP usage (mM/s)')
ax= axs_flux[1] 
ax.plot(sol.t, bioenergetic_model.phi_ck(sol.y[0,], sol.y[1,]))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Flux CK (mM/s)')
ax= axs_flux[2] 
ax.plot(sol.t, bioenergetic_model.phi_oxphos(bioenergetic_model.c_a_tot - sol.y[0,]))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Flux OXPHOS (mM/s)')

#######################
# Compute the time constants from the data 
total_energy = cumtrapz(E_tot + q_r, t_vec, initial = 0) * energy_unit_scaler
total_energy_crop = total_energy[t_vec > params['cycle_length'] * params['N_cycles']]
t_vec_crop = t_vec[t_vec > params['cycle_length'] * params['N_cycles']]
def expfun(x,a,b): 
    return  a * (1 - np.exp(-x/b))
opt_res = curve_fit(expfun, t_vec_crop, total_energy_crop)
exp_prams = opt_res[0]
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec_crop, total_energy_crop) 
ax.plot(t_vec_crop, expfun(t_vec_crop, *exp_prams))
print(f'Time constant: {exp_prams[1]}s')