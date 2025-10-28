'''
This code implements the model as designed in Kushmerick 1999, and adapted in Vicini 2000

This version of the code has been adapted such that the initial resting rate of the model will altered based of Vmax. This allows for a fixed c_pcr_0 while othere parameters are varied... I do not believe it is expected that if rate constants vary then there shoulld be an alteratation in steady-state metabolism

Ryan Konno
'''

# Import 
import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt 
import matplotlib.cm as cmap
import itertools
import sys 
sys.path.append('./')

# Parameters 
params = {
    # Time parameters
    't_start': 0, # s
    't_end': 1400, # s, 

    # Energetics parameters 
    # 'Energetics_model':{

        'c_c_tot': 42, # mM, Harris 1974
        'c_atp_0': 8.2, # mM, Harris 1974
        'c_pcr_0': 32, # mM, Approximate value Vicini 2000
        # 'c_c_tot': 20, # mM, Grassi 1998
        # 'c_atp_0': 6.5, # mM, Grassi 1998
        'Pi_0': 3.183, # mM, Vicini 2000

        'V_max_oxphos': 0.5, # mM/s, Vicini 2000... TBD (may need to optimise for this parameter)
        # 'V_max_oxphos': 14.8 / 60, # mM/s, Vicini 2000... TBD (may need to optimise for this parameter)

        'k_rest': 0.0014, # 1/s, Vicini 2000, estimated off of experimental data Blei et al. 1993
        'k_stim': 0.0139,  # 1/s, Vicini 2000, estimated from exp data Blei et al. 1993
        'k_post': 0.9 * 0.0139,  # 1/s, Vicini 2000, estimated from exp data Blei et al. 1993 NOTE: cannot find value for this rate... assume half of stim?

        'K_adp': 0.058, # mM, Vicini 2000.... TBD (may need to optimise for this parameter)
        'nh': 2.57, # unitless, VIcini 2000, .... TBD (may need to optimise for this parameter)

        'V_ck_f': 100, # mM/s, Kushmerick 1998
        'K_b': 1.11, #mM, MacFarland 1994
        'K_ia': 0.135, # mM, MacFarland 1994
        'K_eq': 1.77e2, # ?, Assuming a pH of 7, Lawson 1979
        'K_iq': 3.5, # mM, MacFarland 1994
        'K_ib': 3.9, # mM, MacFarland 1994
        'K_p': 3.8, # mM, MacFarland 1994

        # Energetic constant to predict energetic rates
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
    def __init__(self, params): 

        # Initialize the main parameters

        self.c_c_tot = params['c_c_tot']
        self.c_pcr_0 = params['c_pcr_0']

        # Initialise maximum rates 
        self.V_max_oxphos = params['V_max_oxphos']

        # ATP Consumption rates 
        self.k_rest = params['k_rest']
        self.k_stim = params['k_stim']
        self.k_post = params['k_post']

        self.K_adp = params['K_adp']
        self.nh = params['nh']

        self.V_ck_f =  params['V_ck_f'] 
        self.K_b = params['K_b'] 
        self.K_ia = params['K_ia'] 
        self.K_eq = params['K_eq'] 
        self.K_iq = params['K_iq'] 
        self.K_ib = params['K_ib'] 
        self.K_p = params['K_p'] 
        
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
         

    def phi_atp(self, t, c_atp): 
        ''' 
        Function defining ATP use during the contraction 
        
        We assume a constant stimulation of 12s 
        '''
        # atp_peak = 0.213 # umol/s/(g wet wt) [Based on initial rates from Phillips et al. 1993]
        # atp_peak = self.k_stim # mM/s, from Vicini 2000
        # t_stim_start = 100
        # t_stim_end = 150
        # # trampend = 1
        # return self.phi_rest * (t < t_stim_start) + atp_peak * (t >= t_stim_start) * (t < t_stim_end) + self.phi_rest * (t >= t_stim_end)
    
        # Adapted for Vicini/Blei experiment 
        k_rest = self.k_rest 
        k_stim = self.k_stim    
        k_post = self.k_post 
        tA = 0 
        tB = 180 
        tB1 = 480 
        tC = 540 
        tD = 666 
        tE = 757 
        return c_atp * (
            k_rest * (t <= tC) + \
            k_stim * (t > tC) * (t <= tD) + \
            k_post * (t > tD) * (t <= tE) + \
            k_rest * (t > tE)
        )


    
        # atp_peak * (t < tstimend) + atp_peak * 0.5 * (np.sin((np.pi*(t-tstimend) / (trampend-tstimend) + np.pi/2)) + 1) * (t > tstimend) * (t < trampend) + self.phi_rest
        # return 50 * (t%10 < 5) 

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
        print('Solving bioenergetics...')

        # Calculate the ICs assuming that c_pcr_0 is known
        c_adp_0 = c_atp_0 * (self.c_c_tot - self.c_pcr_0) / (self.K_eq * self.c_pcr_0)
        self.k_rest = self.V_max_oxphos * (c_adp_0 / self.K_adp)**self.nh / c_atp_0 / (1 +(c_adp_0 / self.K_adp)**self.nh) 

        # Define resting rate 
        self.phi_rest = self.k_rest * c_atp_0

        # Initial conditions 
        # c_pcr_0 calculated as in eqn 11 Vicini 2000 
        c_pcr_0 = (c_atp_0 * self.c_c_tot) / (c_atp_0 + self.K_eq * c_adp_0)
        
        y_0 = (c_atp_0, c_pcr_0)

        self.c_a_tot = c_atp_0 + c_adp_0

        sol = solve_ivp(self.rhs, t_span, y_0, "BDF", max_step = 0.1)
        
        return sol

        
# Define and solve the model 
t_span = (params['t_start'],params['t_end']) 
c_atp_0 = params['c_atp_0']

model = Bioenergetics(params) 

sol = model.solveBioenergetics(t_span, c_atp_0)

# Plot the model output 
fig, axs = plt.subplots(1,2, layout = 'constrained', figsize = (10, 4))
ax = axs[0] 
ax.plot(sol.t, sol.y[0,], label = 'ATP')
# ax.plot(sol.t, model.phi_atp(sol.t), label = 'ATP Consumption', ls = ':')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Concentration of ATP (mM)')
ax= axs[1] 
ax.plot(sol.t, sol.y[1,], label = 'PCr')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Concentration of PCr (mM)')
plt.show()

# Plot the model fluxes 
fig, axs = plt.subplots(1,3, layout = 'constrained', figsize = (10, 4))
ax = axs[0] 
ax.plot(sol.t, model.phi_atp(sol.t, sol.y[0,]))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Flux ATP usage (mM/s)')
ax= axs[1] 
ax.plot(sol.t, model.phi_ck(sol.y[0,], sol.y[1,]))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Flux CK (mM/s)')
ax= axs[2] 
ax.plot(sol.t, model.phi_oxphos(model.c_a_tot - sol.y[0,]))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Flux OXPHOS (mM/s)')

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


