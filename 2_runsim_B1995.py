'''
This is a code to simulate bioenergetics of muscle contraction from the conditions in Barclay et al., 1995

This version does not run the optimisation of the code, but just the simulation with given parameters (see barclay1995OptimiseParams.py for the code with the optimisation)

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
            'V_max_oxphos': 3.47, # mM/s, Vicini 2000... TBD
            # 'V_max_oxphos':  0.5, # mM/s, Vicini 2000... TBD
            'atp_peak': 0.25,# 0.213, # mM/s Peak atp rate calculated based on initial heat rate and enthalpy of ATP from Phillips et al. 1993

            'gamma': 1, # Scaling factor for metabolic rates at rest

            # May need to tune these parameters...
            'K_adp': 0.0615, # mM, Vicini 2000.... TBD (may need to optimise for this parameter)
            # 'K_adp': 0.05, # mM, Vicini 2000.... TBD (may need to optimise for this parameter)
            'nh': 0.873, # unitless, V/Icini 2000, .... TBD (may need to optimise for this parameter)
            # 'nh': 3, # unitless, Varied

            # Values from Barclay et al. 1995
            'F_0': 0, # N, 
            'l_0':  9.5e-3, # m, 
            'mass': 1.99e-3, # g, 
           
            'r_rec': 2.41e5, # F0 l0 / mol, Optimised value
            
            # Heat data used for optimisation 
            'heat_exp_rec': np.array((5.339988e-03, 5.726164e-02, 7.235249e-02, 7.009909e-02)), # J/F0l0, Slow, recovery heat 
        }, 
        'EDL': { 
            'c_c_tot': 29.5, # mM, Kushmerick et al. 1992 
            'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992 
            'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992 
            'Pi_0': 0, # mM,  Kushmerick et al. 1992 
            'V_max_oxphos': 3.47/2, # mM/s, Vicini 2000... TBD
            # For Phillips Simulation 
            'atp_peak': 0.25,# 0.213, # mM/s Peak atp rate calculated based on initial heat rate and enthalpy of ATP from Phillips et al. 1993

            'gamma': 1, # Scaling factor for metabolic rates at rest

            # May need to tune these parameters...
            'K_adp': 0.0615, # mM, Vicini 2000.... TBD (may need to optimise for this parameter)
            'nh': 0.873, # unitless, VIcini 2000, .... TBD (may need to optimise for this parameter)
            # 'nh': 1.180, # unitless, Optimised to B1995 dataset 
            # 'nh': 1, # unitless, Varied

            'F_0': 0, # N, 
            'l_0': 8.9e-3, # m,
            'mass': 3.9e-3, # g, 
            
            # 'r_rec': 1 / 0.8 * 60e3, # J / mol, Assumes mitochondrial efficiency based on average mouse sol and edl
            'r_rec': 2.41e5, # J / mol, Optimised value
            
            # Heat data used for optimisation 
            'heat_exp_rec': np.array((2.518854e-02, 6.404227e-02, 6.903081e-02, 7.313249e-02)), # J/F0l0, Fast, recovery heat 
        },

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

# cOMPUTE the scaled initial energy 
e_init_scale = (params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] / params[params['muscle']]['mass'])**-1
E_tot_scaled = E_tot * e_init_scale # Units of F0l0 / s

#########
# Define the code for the optimisation 
muscle = params['muscle']


# Define the recovery heat from the experimental data 
cycle = np.array((1,5,15,30)) # Cycle numbers
t_exp = params['t_cycle_start'] +  cycle * 5 # s, Times for teh experimental values 

from scipy.interpolate import PchipInterpolator
def rec_heat_exp(t, heat_exp_rec): 
    cspline_exp = PchipInterpolator(t_exp, heat_exp_rec)
    return cspline_exp(t) * (t >= params['t_cycle_start'] + 5) * (t < params['t_cycle_end'])

# Rerun the model with the optimal values 
bioenergetic_model = Bioenergetics(params) 
# Solve the model
sol = bioenergetic_model.solveBioenergetics(t_span, c_atp_0, t_vec, E_tot)
# Compute the energetic rates 
# scale =  params[params['muscle']]['mass'] / params[params['muscle']]['F_0'] / params[params['muscle']]['l_0'] 
# scale =  params[params['muscle']]['mass'] 
scale = 1
q_r = bioenergetic_model.computeRecoveryEnergetics(sol.t, sol.y[0,]) * scale # In units of W/F0l0

# Plot with units in F0l0/s
# Plot the rates 
fig, ax = plt.subplots(layout = 'constrained')
# e_init_scale = (params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] / params[params['muscle']]['mass'])
# E_tot = E_tot * e_init_scale
# energy_unit_scaler = params[params['muscle']]['F_0'] * params[params['muscle']]['l_0'] / params[params['muscle']]['mass'] * 1e3 # convert from W/F0l0 to mW/g 
energy_unit_scaler = 1 
ax.plot(t_vec, E_tot_scaled * energy_unit_scaler , label = '$\dot e_{init}$') 
ax.plot(t_vec, q_r * energy_unit_scaler, label = '$\dot q_r$') 
ax.plot(t_vec, (E_tot_scaled + q_r) * energy_unit_scaler, label = '$\dot q_r + \dot e_{init}$') 
ax.legend()
ax.set_xlabel('Time (s)')
# ax.set_ylabel('Energy rate ($mW g^{-1}$)')
ax.set_ylabel('Energy rate ($F_0 l_0 s^{-1}$)')
# Plot the total energy over the cycle
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, cumtrapz(E_tot_scaled, t_vec, initial = 0) * energy_unit_scaler, label = '$ e_{init}$') 
ax.plot(t_vec, cumtrapz(q_r, t_vec, initial = 0) * energy_unit_scaler, label = '$ q_r$') 
ax.plot(t_vec, cumtrapz(E_tot_scaled + q_r, t_vec, initial = 0) * energy_unit_scaler, label = '$ q_r + e_{init}$')
ax.legend()
ax.set_xlabel('Time (s)')
# ax.set_ylabel('Energy  ($mJ g^{-1}$)')
ax.set_ylabel('Energy  ($F_0 l_0$)')

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
ax.set_title(f'Exponential Decay Fit ({muscle})')
ax.legend()
ax.grid(True)

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
