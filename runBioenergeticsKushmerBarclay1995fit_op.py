'''
This code implements the model as designed in Kushmerick 1999, and adapted in Vicini 2000

This version of the code has been adapted such that the initial resting rate of the model will altered based of Vmax. This allows for a fixed c_pcr_0 while othere parameters are varied... I do not believe it is expected that if rate constants vary then there shoulld be an alteratation in steady-state metabolism

This code fits the recovery heat parameter + ATP Pcr mdoel parameters to the Barclay 1995 dataset

Ryan Konno
'''

# Import 
import numpy as np 
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt 
import matplotlib.cm as cmap
import itertools
import sys 
sys.path.append('./')
# Parameters
params = {
    # Time parameters
    't_start': 0, # s
    't_end': 150, # s
    't_cycle_start': 20, # s
    't_cycle_end': 150, # s 

    # Energetics parameters 
    # 'Energetics_model':{

    'muscle': 'SOL', # Specify muscle parameters to be used in simultation

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
            'atp_peak': 0.25,# 0.213, # mM/s Peak atp rate calculated based on initial heat rate and enthalpy of ATP from Phillips et al. 1993

            'F_0': 0.041, # N, 
            'l_0': 9.5e-3, # m, 
            'mass': 1.99e-3, # g, 
           
            'r_rec': 1 / 0.8 * 60e3 # J / mol, Assumes mitochondrial efficiency based on average mouse sol and edl


        }, 
        'EDL': { 
            'c_c_tot': 29.5, # mM, Kushmerick et al. 1992 
            'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992 
            'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992 
            'Pi_0': 0, # mM,  Kushmerick et al. 1992 
            'V_max_oxphos': 1.88/2, # mM/s, Vicini 2000... TBD
            # For Phillips Simulation 
            'atp_peak': 0.25,# 0.213, # mM/s Peak atp rate calculated based on initial heat rate and enthalpy of ATP from Phillips et al. 1993

            'F_0': 0.057, # N, 
            'l_0': 10e-3, # m,
            'mass': 2.85e-3, # g, 
            
            'r_rec': 1 / 0.8 * 60e3 # J / mol, Assumes mitochondrial efficiency based on average mouse sol and edl

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

        # Energetic constant to predict energetic rates 
        # 'r_rec': 1 / 0.8 * 60e3 # J / mol, Assumes mitochondrial efficiency based on average mouse sol and edl
    # }
}

#### 
# Import experimental data to fit the model
# def f(x, a, b, c, d): 
#     # x is the cycle number
#     return a * b ** x - c
cycle = np.array((1,5,15,30)) # Cycle numbers
t_exp = params['t_cycle_start'] +  cycle * 5 # s, Times for teh experimental values 
heat_exp_rec = np.array((5.339988e-03, 5.726164e-02, 7.235249e-02, 7.009909e-02)) # J/F0l0, Slow, recovery heat 
# heat_exp_rec = np.array((2.518854e-02, 6.404227e-02, 6.903081e-02, 7.313249e-02)) # J/F0l0, Fast, recovery heat 
def rec_heat_exp(t): 
    cspline_exp = PchipInterpolator(t_exp, heat_exp_rec)
    return cspline_exp(t) * (t >= params['t_cycle_start'] + 5) * (t < params['t_cycle_end'])
# cspline_exp = PchipInterpolator(t_exp, heat_exp_rec)
fig, ax = plt.subplots(layout = 'constrained')
ax.plot(np.linspace(0,50*5, 100), rec_heat_exp(np.linspace(0,50*5, 100))) 
ax.plot(t_exp, heat_exp_rec, '.') 
plt.show()


### 
# Create a class for the ode solver 
class Bioenergetics(): 
    def __init__(self, params): 
        
        # Define the muscle 
        self.muscle = params['muscle'] 

        # Define stimulation parameters 
        self.t_cycle_start = params['t_cycle_start']
        self.t_cycle_end = params['t_cycle_end']

        # Initialize the main parameters

        self.c_c_tot = params[self.muscle]['c_c_tot']
        self.c_pcr_0 = params[self.muscle]['c_pcr_0']

        # Initialise maximum rates 
        self.V_max_oxphos = params[self.muscle]['V_max_oxphos']

        # ATP Consumption rates 
        self.k_rest = params['k_rest']
        self.k_stim = params['k_stim']
        self.k_post = params['k_post']
        self.atp_peak = params[self.muscle]['atp_peak']

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
        self.r_rec = params[self.muscle]['r_rec']
        
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
        t_start_cycle = self.t_cycle_start
        t_end_cycle = self.t_cycle_end

        # Normalize time
        t_cycle_length = 5 # s, Length of the cycle
        t_cycle = t%t_cycle_length

        
        if muscle == 'SOL': 
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
        elif muscle == 'EDL':
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

        return  self.k_rest * c_atp *  (t <= t_start_cycle)\
                 + (self.k_rest * c_atp +( atp_peak) * (t_cycle < tstimend) + atp_peak * 0.5 * (np.sin((np.pi*(t_cycle-tstimend) / (trampend-tstimend) + np.pi/2)) + 1) * (t_cycle > tstimend) * (t_cycle < trampend)) * (t > t_start_cycle) * (t <= t_end_cycle)\
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

        # Here we assume self.r_rec is in units of J / mol
        # Converted phi_oxphos from umol/s/g to mol/s/g
        energy_rate = self.r_rec * self.phi_oxphos(c_adp_) * 1e-6 

        return energy_rate # J / g / s



# First perform an optimisation of only the recovery heat parameters 

def f_opt(x):
    '''
    Function to optimize
        Input: 
            x - parameter vector 
    '''
    ####
    # Define the bioenergetics model 
    model = Bioenergetics(params)
    # model.K_adp = x[0]
    # model.V_max_oxphos = x[1]
    # model.atp_peak = x[2]
    #####
    # Solve the ode 
    t_span = (params['t_start'],params['t_end']) 
    sol = model.solveBioenergetics(t_span, params[params['muscle']]['c_atp_0'])
    # PCr, ADP, activation = sol.T  # Transpose to get individual variables
    c_atp, c_pcr = sol.y  # Transpose to get individual variables
    # ####
    # # Error calculation 
    # # Interpolate values to compare 
    # pcr_model = np.interp(t_exp, sol.t, c_pcr)

    # Compute the recovery  
    model.r_rec = x
    recovery_heat_model_ = model.computeRecoveryEnergetics(c_atp)

    # Scale the recovery heat to units of F0 l0 / s
    scale_factor = params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0']
    recovery_heat_model = recovery_heat_model_ * scale_factor

    # compute the recovery heat from the experiment 
    recovery_heat_exp = rec_heat_exp(sol.t) # Units of F0l0/s

    # only fit over the times from t_cycle_start to t_cycle_end 
    recovery_heat_model_crop = recovery_heat_model[(sol.t > params['t_cycle_start'] + 5) * (sol.t < params['t_cycle_end'])]
    recovery_heat_exp_crop = recovery_heat_exp[(sol.t > params['t_cycle_start']+ 5) * (sol.t < params['t_cycle_end'])]
    
    # fig, ax = plt.subplots(layout = 'constrained')
    # ax.plot(recovery_heat_model_crop)
    # ax.plot(recovery_heat_exp_crop)
    # plt.show()
    # Compute the error 
    error = np.linalg.norm(recovery_heat_model_crop - recovery_heat_exp_crop) / np.linalg.norm(recovery_heat_exp_crop)

    return error 

# initialise experimental parameters
t_span = (params['t_start'],params['t_end']) 

# muscle = 'SOL'
# Perform optimization 
for muscle in ('SOL', 'EDL'):
    params['muscle'] = muscle
    x0 = (params[muscle]['r_rec'])
    constraints = (5e4,2e6)
    opt_res = minimize_scalar(f_opt, constraints, constraints, method = 'Bounded', options = {'disp': False, 'maxiter': 500})
    # opt_res = minimize(f_opt, x0, method = 'Nelder-Mead', bounds = constraints, options = {'disp': True, 'maxiter': 500})
    print(f'Optimal parameters for {muscle}: {opt_res.x}')
     
    # Update the value in the params file 
    params[muscle]['r_rec'] = opt_res.x
    model = Bioenergetics(params)
    sol = model.solveBioenergetics(t_span, params[params['muscle']]['c_atp_0'])
    # Compute the efficiency based on substrates + thermodynamics theory 
    n_atp = 38 # number of atp
    Gatp = 60e3 # J/mol 
    # r_rec is in units of J / mol
    r_rec_J = params[muscle]['r_rec'] 
    efficiency = n_atp * Gatp / (params[muscle]['r_rec'] + n_atp * Gatp)
    print(f'    efficiency: {efficiency}')

    # Plot the recovery heat for verification 
    fig, ax = plt.subplots(layout = 'constrained')
    recovery_heat_model_ = model.computeRecoveryEnergetics(sol.y[0,])
    scale_factor = params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0']
    recovery_heat_model = recovery_heat_model_ * scale_factor # Now in F0l0/s
    ax.plot(sol.t, recovery_heat_model, label = 'Mod rec')
    ax.plot(sol.t, rec_heat_exp(sol.t), ls = ':', label = 'Exp rec')
    ax.plot(sol.t, model.phi_atp(sol.t, sol.y[0,]) * 10**-6 * scale_factor * Gatp, label = "Init" )
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (F0l0/s)')    
    plt.show()





fig_conc, axs_conc = plt.subplots(1,2, layout = 'constrained', figsize = (10, 4))
fig_flux, axs_flux = plt.subplots(1,3, layout = 'constrained', figsize = (10, 4))
fig_ener, axs_ener = plt.subplots(1,2, layout = 'constrained', figsize = (10, 4))

for muscle in ('SOL', 'EDL'):
    # Set the muscle 
    params['muscle'] = muscle 

    # Define and solve the model 
    t_span = (params['t_start'],params['t_end']) 
    c_atp_0 = params[muscle]['c_atp_0']

    model = Bioenergetics(params) 

    sol = model.solveBioenergetics(t_span, c_atp_0)

    # Plot the model output 
    ax = axs_conc[0] 
    ax.plot(sol.t, sol.y[0,], label = 'ATP')
    # ax.plot(sol.t, model.phi_atp(sol.t), label = 'ATP Consumption', ls = ':')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Concentration of ATP (mM)')
    ax= axs_conc[1]
    ax.plot(sol.t, sol.y[1,] - model.c_pcr_0, label = muscle + 'PCr')
    # ax.plot(t_exp, pcr_exp, '.', label = 'PCr_Exp')
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Concentration of PCr (mM)')


    # Plot the model fluxes 
    ax = axs_flux[0] 
    ax.plot(sol.t, model.phi_atp(sol.t, sol.y[0,]))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flux ATP usage (mM/s)')
    ax= axs_flux[1] 
    ax.plot(sol.t, model.phi_ck(sol.y[0,], sol.y[1,]))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flux CK (mM/s)')
    ax= axs_flux[2] 
    ax.plot(sol.t, model.phi_oxphos(model.c_a_tot - sol.y[0,]))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flux OXPHOS (mM/s)')

    # Plot the energetic rates 
    if muscle == 'SOL': 
        ax = axs_ener[0]
        ax.set_title('SOL')
        def f(x, a, b, c, d): 
            ''' 
            Function as defined in computeParametersBarclay1995.py 
            *** Ensure its the same if any adjustments are made (e.g. fibre-type) ***
            '''
            return a * np.exp(b * x - c) + d
        popt = np.array((26.31884038, -0.42107639,  4.45702316,  0.54965355))
        # ax.plot(sol.t, model.phi_atp(sol.t, sol.y[0,]), label = 'init')
    elif muscle == 'EDL':
        ax = axs_ener[1]
        ax.set_title('EDL')
        def f(x, a, b, c, d): 
            ''' 
            Function as defined in computeParametersBarclay1995.py 
            *** Ensure its the same if any adjustments are made (e.g. fibre-type) ***
            '''
            return a * b ** x - c
        popt = np.array((0.3194931, 0.63699497, -2.01981868, 1.)) 

    # Compute scale to go from J/s/g to F0l0/s
    # the 1e-3 comes from scaling from mM to M
    scale =  params[muscle]['mass'] / params[muscle]['F_0'] / params[muscle]['l_0'] 
    # Plot initial
    ax.plot(sol.t, model.phi_atp(sol.t, sol.y[0,]) * 60e3 * 1e-6 * scale , label = 'init')
    # Plot recovery 
    ax.plot(sol.t, model.computeRecoveryEnergetics(sol.y[0,]) * scale , label = 'rec')      
    ax.set_xlabel('Time (s) ')
    ax.set_ylabel('Energetic rate (F0l0/s)')
    ax.legend()

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



