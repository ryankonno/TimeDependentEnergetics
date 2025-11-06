'''
This code implements the model as designed in Kushmerick 1999, and adapted in Vicini 2000

This version of the code has been adapted such that the initial resting rate of the model will altered based of Vmax. This allows for a fixed c_pcr_0 while othere parameters are varied... I do not believe it is expected that if rate constants vary then there shoulld be an alteratation in steady-state metabolism

Ryan Konno
'''

# Import 
import numpy as np 
from scipy.integrate import solve_ivp, cumulative_trapezoid
import matplotlib.pyplot as plt 
import matplotlib.cm as cmap
import itertools
import sys 
sys.path.append('./')
# Parameters
params = {
    # Time parameters
    't_start': 0, # s
    't_end': 400, # s

    # Energetics parameters 
    # 'Energetics_model':{

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
            'V_max_oxphos':  1.88, # mM/s, Opt to Phillips
            # 'V_max_oxphos':  5, # mM/s, Vicini 2000... TBD
            # For Phillips Simulation 
            'atp_peak': 0.25,# 0.213, # mM/s Peak atp rate calculated based on initial heat rate and enthalpy of ATP from Phillips et al. 1993

            'F_0': 0.0102, # N, 
            'l_0': 9.5e-3, # m,
            'mass': 1.99e-3, # g, 
            # Energetic constant to predict energetic rates 
            # 'r_rec': (1 - 0.76) / 0.76 * 32 * 60e3, # J/ s / L, Assumes mitochondrial efficiency based on average mouse sol 
            # 'r_rec': (1 - 0.76) * 2810e3, # J/ s / L, Assumes mitochondrial efficiency based on average mouse sol 
            'r_rec': 1 / 0.86 * 60e3, # J/ s / L, Assumes mitochondrial efficiency based on average mouse sol 

            'duty_cycle':  0.16, # unitless, Duty cycle of initial heat rate

        }, 
        'EDL': { 
            'c_c_tot': 29.5, # mM, Kushmerick et al. 1992 
            'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992 
            'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992 
            'Pi_0': 0, # mM,  Kushmerick et al. 1992 
            'V_max_oxphos': 1.88/2, # mM/s, Opt to Phillips
            # 'V_max_oxphos':  2.5, # mM/s, Vicini 2000... TBD
            # For Phillips Simulation 
            'atp_peak': 0.25,# 0.213, # mM/s Peak atp rate calculated based on initial heat rate and enthalpy of ATP from Phillips et al. 1993

            'F_0': 0.0102, # N, 
            'l_0': 10e-3, # m,
            'mass': 2.85e-3, # g, 

            # Energetic constant to predict energetic rates 
            # 'r_rec': (1 - 0.86) / 0.86 * 32 * 60e3, # J/ s / L, Assumes mitochondrial efficiency based on average mouse sol 
            # 'r_rec': (1 - 0.86) * 2810e3, # J/ s / L, Assumes mitochondrial efficiency based on average mouse sol 
            'r_rec': 1 / 0.86 * 60e3, # J/ s / L, Assumes mitochondrial efficiency based on average mouse sol 

            'duty_cycle':  0.04, # unitless, Duty cycle of initial heat rate
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

        # May need to tune these parameters...
        'K_adp': 0.058, # mM, Vicini 2000.... TBD (may need to optimise for this parameter)
        'nh': 2.57, # unitless, VIcini 2000, .... TBD (may need to optimise for this parameter)

        # Assume constant across all species and muscle fibre-types
        'V_ck_f': 1000,# 100, # mM/s, Kushmerick 1998
        'K_b': 1.11, #mM, MacFarland 1994
        'K_ia': 0.135, # mM, MacFarland 1994
        'K_eq': 1.77e2, # ?, Assuming a pH of 7, Lawson 1979
        'K_iq': 3.5, # mM, MacFarland 1994
        'K_ib': 3.9, # mM, MacFarland 1994
        'K_p': 3.8, # mM, MacFarland 1994


    # }
}

#### 
# Import experimental data to fit the model
# #       NOTE: Experimental data only had a 4s time resolution
# datapath = 'Data/PhillipsData_dPCR.csv'
# exp_data = np.loadtxt(datapath, delimiter=",", skiprows=1)
# # Scale the data so that pcr is negative 
# t_exp = exp_data[:,0] # Units of s
# pcr_exp = -exp_data[:,1] # Converted to units of umol/(g wet wt)

### 
# Create a class for the ode solver 
class Bioenergetics(): 
    def __init__(self, params, muscle = 'SOL', freq = 1/5): 
        
        # Define the muscle 
        self.muscle = muscle 
        self.t_cycle = 1/freq

        # Initialize the main parameters

        self.c_c_tot = params[muscle]['c_c_tot']
        self.c_pcr_0 = params[muscle]['c_pcr_0']

        # Initialise maximum rates 
        self.V_max_oxphos = params[muscle]['V_max_oxphos']

        # ATP Consumption rates 
        self.k_rest = params['k_rest']
        self.k_stim = params['k_stim']
        self.k_post = params['k_post']
        self.atp_peak = params[muscle]['atp_peak']

        self.K_adp = params['K_adp']
        self.nh = params['nh']

        self.V_ck_f =  params['V_ck_f'] 
        self.K_b = params['K_b'] 
        self.K_ia = params['K_ia'] 
        self.K_eq = params['K_eq'] 
        self.K_iq = params['K_iq'] 
        self.K_ib = params['K_ib'] 
        self.K_p = params['K_p'] 

        # Energetic rate parameter 
        self.r_rec = params[muscle]['r_rec']

        self.duty_cycle = params[muscle]['duty_cycle']
        self.init_heat_scale = 1 # Parameter to scale the initial heat (e.g. implement higher heats with faster frequency)
        
    def c_adp(self, c_atp_):
        return self.c_a_tot - c_atp_
    
    def c_cr(self, c_pcr_): 
        return self.c_c_tot - c_pcr_
    
    def phi_oxphos(self, c_adp_): 
        return self.V_max_oxphos * (c_adp_ / self.K_adp) ** self.nh / (1 + (c_adp_ / self.K_adp)**self.nh)

    def phi_ck(self, c_atp, c_pcr): 
        c_adp = self.c_a_tot - c_atp 
        c_cr = self.c_c_tot - c_pcr
        return self.V_ck_f / self.K_b / self.K_ia * (c_adp * c_pcr - c_cr * c_atp / self.K_eq)/ \
                    (1 + c_adp / self.K_ia + c_atp / self.K_iq + c_pcr / self.K_ib \
                     + c_adp * c_pcr / self.K_b / self.K_ia + c_cr * c_atp / self.K_iq / self.K_p)
         
    # def phi_atp(self, t, c_atp): 
    #     ''' 
    #     Function defining ATP use during the contraction 
        
    #     Modified to simulate the contractions from Barclay et al. 1995
    #     '''
    #     # Constant atp_peak [we use variable atp usage now... see below]
    #     # atp_peak = 0.75 # umol/s/(g wet wt) [computed using computeParametersBarclay1995.py]

    #     tstimend = 0.8 # s, Length of stimulation (B1995)
    #     trampend = 1
    #     t_start_cycle = 20

    #     # Normalize time
    #     t_cycle_length = 5 # s, Length of the cycle
    #     t_cycle = t%t_cycle_length

    #     # Use variable ATP usage 
    #     def f(x, a, b, c, d): 
    #         ''' 
    #         Function as defined in computeParametersBarclay1995.py 
    #         *** Ensure its the same if any adjustments are made (e.g. fibre-type) ***
    #         '''
    #         return a * np.exp(b * x - c) + d
    #     popt = np.array((26.31884038, -0.42107639,  4.45702316,  0.54965355))
    #     cycle_count = np.floor(t/t_cycle_length) + 1
    #     # Compute the atp usage based on the cycle number
    #     atp_peak = f(cycle_count, *popt) # umol/s/(g wet wt) [computed using computeParametersBarclay1995.py]

    #     return (atp_peak * (t_cycle < tstimend) + atp_peak * 0.5 * (np.sin((np.pi*(t_cycle-tstimend) / (trampend-tstimend) + np.pi/2)) + 1) * (t_cycle > tstimend) * (t_cycle < trampend)) * (t > t_start_cycle) + self.k_rest * (t <= t_start_cycle)
    
    def phi_atp(self, t, c_atp): 
        ''' 
        Function defining ATP use during the contraction 
        
        Modified to simulate the contractions from Barclay et al. 1995
        '''
        # Constant atp_peak [we use variable atp usage now... see below]
        # atp_peak = 0.75 # umol/s/(g wet wt) [computed using computeParametersBarclay1995.py]

        
        trampend = 1
        t_start_cycle = 20
        t_end_cycle = 200


        # Normalize time
        t_cycle_length = self.t_cycle # s, Length of the cycle
        t_cycle = t%t_cycle_length

        
        if self.muscle == 'SOL': 
            # Use variable ATP usage 
            def f(x, a, b, c, d): 
                return a * np.exp(b * x - c) + d
            popt = np.array((26.31884038, -0.42107639,  4.45702316,  0.54965355))
            tstimend = self.duty_cycle * t_cycle_length # s, Length of stimulation (B1995)

        elif self.muscle == 'EDL':
            # Use variable ATP usage 
            def f(x, a, b, c, d): 
                return a * b ** x - c
            popt = np.array((0.3194931, 0.63699497, -2.01981868, 1.)) 
            tstimend = self.duty_cycle * t_cycle_length # s, Length of stimulation (B1995)

        
        cycle_count = np.floor(t/t_cycle_length) + 1
        # Compute the atp usage based on the cycle number
        # atp_peak = f(cycle_count, *popt) # umol/s/(g wet wt) [computed using computeParametersBarclay1995.py]
        atp_peak = 1
        

        # return  self.k_rest * c_atp *  (t <= t_start_cycle)\
        #          + (self.k_rest * c_atp +( atp_peak) * (t_cycle < tstimend) + atp_peak * 0.5 * (np.sin((np.pi*(t_cycle-tstimend) / (trampend-tstimend) + np.pi/2)) + 1) * (t_cycle > tstimend) * (t_cycle < trampend)) * (t > t_start_cycle) * (t <= t_end_cycle)\
        #          + self.k_rest * c_atp *  (t > t_end_cycle)
        return  self.k_rest * c_atp *  (t <= t_start_cycle)\
                 + (self.k_rest * c_atp + (self.init_heat_scale *  atp_peak) * (t_cycle < tstimend)) * (t > t_start_cycle) * (t <= t_end_cycle)\
                 + self.k_rest * c_atp *  (t > t_end_cycle)

    def atp_rhs(self, t, y): 
        atp_curr = y[0,]
        adp_curr = self.c_a_tot - atp_curr
        pcr_curr = y[1,]
        return -self.phi_atp(t, atp_curr) + self.phi_oxphos(adp_curr) + self.phi_ck(atp_curr, pcr_curr)
    
    def pcr_rhs(self, t, y): 
        atp_curr = y[0,]
        pcr_curr = y[1,]
        return - self.phi_ck(atp_curr, pcr_curr)
    
    def rhs(self, t, y): 
        return (self.atp_rhs(t,y), self.pcr_rhs(t,y))
    
    def computePi(self): 
        # TODO: implement
        return 0
    
    def solveBioenergetics(self, t_span, c_atp_0):
        # print('Solving bioenergetics...')
        # print(f'K_adp = {self.K_adp}, V_max_oxphos = {self.V_max_oxphos}')

        # Calculate the ICs assuming that c_pcr_0 is known
        c_adp_0 = c_atp_0 * (self.c_c_tot - self.c_pcr_0) / (self.K_eq * self.c_pcr_0)
        self.k_rest = self.V_max_oxphos * (c_adp_0 / self.K_adp)**self.nh / c_atp_0 / (1 +(c_adp_0 / self.K_adp)**self.nh) 
        # print(f'k_rest = {self.k_rest}')

        # Define resting rate 
        self.phi_rest = self.k_rest * c_atp_0

        # Initial conditions 
        # c_pcr_0 calculated as in eqn 11 Vicini 2000 
        c_pcr_0 = (c_atp_0 * self.c_c_tot) / (c_atp_0 + self.K_eq * c_adp_0)
        
        y_0 = (c_atp_0, c_pcr_0)

        self.c_a_tot = c_atp_0 + c_adp_0

        sol = solve_ivp(self.rhs, t_span, y_0, "BDF", max_step = 0.01)
        
        return sol
    
    def computeRecoveryEnergetics(self, c_atp): 

        c_adp_ = self.c_a_tot - c_atp 

        energy_rate = self.r_rec * self.phi_oxphos(c_adp_)

        return energy_rate 


fig_conc, axs_conc = plt.subplots(1,2, layout = 'constrained', figsize = (10, 4))
fig_flux, axs_flux = plt.subplots(1,3, layout = 'constrained', figsize = (10, 4))
fig_ener, axs_ener = plt.subplots(1,2, layout = 'constrained', figsize = (10, 4))
fig_heat, axs_heat = plt.subplots(1,2, layout = 'constrained', figsize = (10, 4))
fig_ratio, axs_ratio = plt.subplots(1,2, layout = 'constrained', figsize = (10, 4))


# stimulation frequencies to evaluate
# freq_list = (1/2, 1, 2)
freq_list = (1/16, 1/8, 1/4)
# freq_list = (1/4, 1/4, 1/4)
heat_scale_list = (1, 1.5, 2)
# heat_scale_list = (1, 1, 1)
# create a viridis colormap for the frequencies
c_map = cmap.viridis(np.linspace(0, 1, len(freq_list)))

for muscle in ("SOL", "EDL"):
    print(f'Muscle:  {muscle}', muscle)
    for i, freq in enumerate(freq_list):
        print(f'    freq = {freq}')
        # Define and solve the model 
        t_span = (params['t_start'], params['t_end']) 
        c_atp_0 = params[muscle]['c_atp_0']

        model = Bioenergetics(params, muscle, freq) 
        model.init_heat_scale = heat_scale_list[i]

        sol = model.solveBioenergetics(t_span, c_atp_0)
        color = c_map[i]

        # Plot the model output 
        ax = axs_conc[0] 
        if muscle == "SOL":
            ax.plot(sol.t, sol.y[0,], label = f'ATP {muscle} f={freq}', color = color)
        # ax.plot(sol.t, model.phi_atp(sol.t), label = 'ATP Consumption', ls = ':')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Concentration of ATP (mM)')

        ax = axs_conc[1]
        if muscle == "SOL": 
            ax.plot(sol.t, sol.y[1,] - model.c_pcr_0, label = f'{muscle} f={freq} PCr', color = color)
        # elif muscle == "EDL": 
        #     ax.plot(sol.t, sol.y[1,] - model.c_pcr_0, label = f'{muscle} f={freq} PCr', color = color, ls = ':')
        # ax.plot(t_exp, pcr_exp, '.', label = 'PCr_Exp')
        ax.legend()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Concentration of PCr (mM)')

        # Plot the model fluxes 
        ax = axs_flux[0] 
        ax.plot(sol.t, model.phi_atp(sol.t, sol.y[0,]), color = color, label = f'{muscle} f={freq}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Flux ATP usage (mM/s)')

        ax = axs_flux[1] 
        ax.plot(sol.t, model.phi_ck(sol.y[0,], sol.y[1,]), color = color)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Flux CK (mM/s)')

        ax = axs_flux[2] 
        ax.plot(sol.t, model.phi_oxphos(model.c_a_tot - sol.y[0,]), color = color)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Flux OXPHOS (mM/s)')

        # Plot the energetic rates 
        if muscle == 'SOL': 
            ax = axs_ener[0]
            ax.set_title('SOL')
            def f(x, a, b, c, d): 
                return a * np.exp(b * x - c) + d
            popt = np.array((26.31884038, -0.42107639,  4.45702316,  0.54965355))
        else:
            ax = axs_ener[1]
            ax.set_title('EDL')
            def f(x, a, b, c, d): 
                return a * b ** x - c
            popt = np.array((0.3194931, 0.63699497, -2.01981868, 1.)) 
             
        # Compute scale to go from J/s/g to F0l0/s
        # the 1e-3 comes from scaling from mM to M
        scale = params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0'] * 1e-3

        # Plot recovery 
        recovery_energy = model.computeRecoveryEnergetics(sol.y[0,]) * 1e-3 * scale 
        ax.plot(sol.t, recovery_energy, label = f'rec, freq = {freq}', color = color)      
        ax.set_xlabel('Time (s) ')
        ax.set_ylabel('Energetic rate (F0l0/s)')
        if muscle == 'EDL': 
            ax.legend()


        # Compute the total heat
        if muscle == 'SOL': 
            ax = axs_heat[0] 
            ax.set_title('SOL')
        else:
            ax = axs_heat[1]
            ax.set_title('EDL')
        heat_tot = cumulative_trapezoid(recovery_energy, sol.t, initial = 0) 
        ax.plot(sol.t, heat_tot, label = f'rec, freq = {freq}', color = color)
        ax.plot(sol.t, cumulative_trapezoid(model.phi_atp(sol.t, sol.y[0,]) * 60 * scale, sol.t, initial = 0) , label = f'init, freq = {freq}',ls = ':', color = color)
        ax.set_xlabel('Time (s) ')
        ax.set_ylabel('Energy (F0l0)')
        if muscle == 'SOL':
            ax.legend()


        #Plot the ratio of initial to recovery heat 
        if muscle == 'SOL': 
            ax = axs_ratio[0] 
            ax.set_title('SOL')
        else:
            ax = axs_ratio[1]
            ax.set_title('EDL')
        init_heat_tot = cumulative_trapezoid(model.phi_atp(sol.t, sol.y[0,]) * 60 * scale, sol.t, initial = 0) 
        rec_heat_tot = cumulative_trapezoid(recovery_energy, sol.t, initial = 0) 
        ax.plot(sol.t, init_heat_tot / rec_heat_tot, color = color)
        ax.set_ylabel('Ratio Init:Rec')
        ax.set_xlabel('Time (s)')
        

        # Compute the integral 
        init_heat_int = np.trapz(sol.t, model.phi_atp(sol.t, sol.y[0,]) * 60 * scale )
        rec_heat_int = np.trapz(sol.t, recovery_energy)
        if freq == freq_list[0]: 
            init_heat_int_0 = init_heat_int
            rec_heat_int_0 = rec_heat_int
        print(f'    Integral of init heat = {init_heat_int}')
        print(f'    Init: Ratio to lowest freq = {init_heat_int / init_heat_int_0}')
        print(f'    Init: Ratio to rec = {init_heat_int / rec_heat_int}')
        print(f'    Integral of rec heat = {rec_heat_int}')
        print(f'    Rec: Ratio to lowest freq = {rec_heat_int / rec_heat_int_0}')
        print(f'    Max recovery heat: {np.max(recovery_energy)}')


plt.show()


# # Perform optimization 
# from scipy.optimize import minimize
# x0 = (0.6, params['Energetics_model']['k_m_1'])
# constraints = ((0,10), (0,10))
# opt_res = minimize(f_opt, x0, method = 'Nelder-Mead', bounds = constraints, options = {'disp': True, 'maxiter': 500})
# print(f'Optimal parameters: {opt_res.x}')

# # Rerun with optimal parameters 
# model = Bioenergetics(*opt_res.x)
# sol = model.solveODE(params['t_start'], params['t_end']) # Solve the IVP 
# PCr, activation = sol.y

# # Plot the comparison between PCr experimental and modelled 
# fig, ax  = plt.subplots(figsize = (6,4), layout = 'constrained') 
# ax.plot(sol.t, PCr, label = 'PCr (model)', color = 'k')
# ax.plot(t_exp, pcr_exp, label = 'PCr (Phillip et al. 1993)', ls = 'None', marker = '.', color = 'k')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('PCr Concentration [mol/g]')
# plt.legend()



