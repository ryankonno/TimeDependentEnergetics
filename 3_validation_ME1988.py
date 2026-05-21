'''
Code the validate the recovery model using data from Mast and Elzinga 1988.

Soleus data is used to inform the model parameters.

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

from Models.ActivationModel import ActivationModel
from Models.MechanicsModel import MechModel 
from Models.InitialEnergeticsModel import EnergeticsModel

# Import parameters 
from parameters_muscle import params as params_muscle

'''
Define protocol specific parmeters 
'''
params_protocol = {
    # Time parameters for setting up the protocol 
    't_start': -10, # s, 10 s pre-stimulation buffer
    't_end': 150, # s
    'cycle_length': 0.3, # s, Defines the length of the cycle (used to set frequency of the contractions)
    'N_cycles': 10, # unitless, Number of cycles to simulate (rest period after N_cycles contractions)
}

# Combine the parameter files 
params = params_protocol | params_muscle

# Define 20deg parameters 
phi_oxmax_val = 1.9322 # Values for soleus and edl
'''
Compute necessary parameters from data 
'''

for muscle in ('SOL',):
    # Maximum isometric force
    params[muscle]['F_0'] = (
        params[muscle]['mass'] / params['rho0'] /
        params[muscle]['l_0'] *
        params[muscle]['max_iso_stress']
    )
    print(f'{muscle}: Maximum isometric force: {params[muscle]["F_0"]}')
        # Reset parameters to 20deg values
    params[muscle]['V_max_oxphos'] = phi_oxmax_val

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
dt = 0.0005
t_vec = np.linspace(params['t_start'], params['t_end'], int((params['t_end'] - params['t_start']) / dt)) 

# Set to use the soleus 
params['muscle'] = 'SOL'

# Compute the stimulation times
stim_protocol_vec, stim_times_vec,  dl_vec = f_stim_length(t_vec, params)
single_run_idx = 0
stim_freq_hz = 0.5

# Containers were previously filled inside a loop over conditions.
component_energy_abs = []
peak_qr_vs_freq = []
efficiency_rows = []

# Ca dynamics
act_model = ActivationModel(params[params['muscle']], t_vec)
idx_stims = np.nonzero(stim_times_vec)[0]
stim_vec, ca_vec, catn_vec = act_model.runExcAct(idx_stims, w_0 = 0.001)
# Plot the results 
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(t_vec, ca_vec, label = 'Free Ca') 
ax.plot(t_vec, catn_vec, label = 'CaTn')
# ax.plot(stim_vec)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Normalised concentration')
fig.savefig(f'Figures/ME1988_Ca_time_{params["muscle"]}.jpg')
fig.savefig(f'Figures/ME1988_Ca_time_{params["muscle"]}.svg')
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
fig_force.savefig(f'Figures/ME1988_Force_{params["muscle"]}.jpg')
fig_force.savefig(f'Figures/ME1988_Force_{params["muscle"]}.svg')

# Compute the initial energetics 
energy_model = EnergeticsModel()
q_a, q_m, q_sl, w = energy_model.solveInitialEnergetics(t_vec, ca_vec, catn_vec, params[muscle], e_ce + 1, dedt_ce, force_direct, mech_model)
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
from Models.BioenergeticsModel import Bioenergetics
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
fig_atp_pcr_conc.savefig(f'Figures/ME1988_ATP_PCR_conc_{params["muscle"]}.jpg')
fig_atp_pcr_conc.savefig(f'Figures/ME1988_ATP_PCR_conc_{params["muscle"]}.svg')

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
fig_atp_pcr.savefig(f'Figures/ME1988_ATP_PCR_deriv_{params["muscle"]}.jpg')
fig_atp_pcr.savefig(f'Figures/ME1988_ATP_PCR_deriv_{params["muscle"]}.svg')

# Compute the energetic rates 
scale =  params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0'] 
q_r = bioenergetic_model.computeRecoveryEnergetics(sol.t, sol.y[0,]) * scale # F0l0/s = J/g/s * g/F0l0

# Compute the scaler to get correct units
energy_unit_scaler = params[muscle]['F_0'] * params[muscle]['l_0'] / params[muscle]['mass'] * 1e3 # convert from F0l0/s to mW/g 

# Plot the total energy over the cycle
fig_energy, ax_energy = plt.subplots(layout = 'constrained')
ax_energy.plot(t_vec, cumtrapz(E_tot, t_vec, initial = 0) * energy_unit_scaler, label = '$ e_{init}$', color = palette[single_run_idx], alpha = 0.25) 
ax_energy.plot(t_vec, cumtrapz(q_r, t_vec, initial = 0) * energy_unit_scaler, label = '$ q_r$', color = palette[single_run_idx], ls = lib.plot_style.ls_styles[2], alpha = 0.5) 
ax_energy.plot(t_vec, cumtrapz(E_tot + q_r, t_vec, initial = 0) * energy_unit_scaler, label = '$ q_r + e_{init}$', color = palette[single_run_idx]) 
ax_energy.set_xlabel('Time (s)')
ax_energy.set_ylabel('Energy  ($mJ g^{-1}$)')
ax_energy.legend()
fig_energy.savefig(f'Figures/ME1988_EnergyUse_{params["muscle"]}.jpg')
fig_energy.savefig(f'Figures/ME1988_EnergyUse_{params["muscle"]}.svg')

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
ax_heat_rate.plot(t_vec, E_tot * energy_unit_scaler, color = palette[single_run_idx], ls = lib.plot_style.ls_styles[2], label = 'Intialheat rate')
ax_heat_rate.plot(t_vec, q_r * energy_unit_scaler, color = palette[single_run_idx], ls = lib.plot_style.ls_styles[1], label = 'Recovery heat rate')
ax_heat_rate.set_xlabel('Time (s)')
ax_heat_rate.set_ylabel('Heat rate ($mW g^{-1}$)')
ax_heat_rate.set_ylim((0,0.75))
ax_heat_rate.legend()
fig_heat_rate.savefig(f'Figures/ME1988_HeatRate_{params["muscle"]}.jpg')
fig_heat_rate.savefig(f'Figures/ME1988_HeatRate_{params["muscle"]}.svg')

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
fig_tau.savefig(f'Figures/ME1988_RecoveryFit_{params["muscle"]}.jpg')
fig_tau.savefig(f'Figures/ME1988_RecoveryFit_{params["muscle"]}.svg')

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