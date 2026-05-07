'''
Bioenergetics model code 

Ryan Konno
r.konno@uq.edu.au
The University of Queensland 
'''

### 
import numpy as np 
from scipy.integrate import solve_ivp, cumtrapz
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt 
font = {'size'   : 14}
plt.rc('font', **font)
import matplotlib.cm as cmap
# Create a class for the ode solver 
class Bioenergetics(): 
    def __init__(self, params): 
        
        # Define the muscle 
        self.muscle = params['muscle'] 

        # Define stimulation parameters 
        # self.t_cycle_start = params['t_cycle_start']
        # self.t_cycle_end = params['t_cycle_end']

        # Initialize the main parameters

        self.c_c_tot = params[self.muscle]['c_c_tot']
        self.c_pcr_0 = params[self.muscle]['c_pcr_0']

        # Initialise maximum rates 
        self.V_max_oxphos = params[self.muscle]['V_max_oxphos']
        self.gamma_1 = params[self.muscle]['gamma']

        # ATP Consumption rates 
        # self.k_rest = params['k_rest']
        # self.k_stim = params['k_stim']
        # self.k_post = params['k_post']
        # self.atp_peak = params[self.muscle]['atp_peak']

        self.K_adp = params[self.muscle]['K_adp']
        self.nh = params[self.muscle]['nh']

        self.V_ck_f =  params['V_ck_f'] 
        self.K_b = params['K_b'] 
        self.K_ia = params['K_ia'] 
        self.K_eq = params['K_eq'] 
        self.K_iq = params['K_iq'] 
        self.K_ib = params['K_ib'] 
        self.K_p = params['K_p'] 

        # Energetic rate parameter 
        self.r_rec = params[self.muscle]['r_rec']

        # Small tolerance used to keep concentrations in a numerically safe range.
        self._eps = 1e-12
        
    def c_adp(self, c_atp_):
        return self.c_a_tot - c_atp_
    
    def c_cr(self, c_pcr_): 
        return self.c_c_tot - c_pcr_
    
    def gamma_fun(self, t, c_adp_, c_atp_):
        # Function to compute gamma 

        # Evaluate phi_atp (may be scalar or array) and form a boolean mask
        phi_atp_val = self.phi_atp(t, c_atp_)
        # print(f'phi_atp_val = {phi_atp_val}')

        # Determine rest phase via concentration of ATP 
        # cond = np.abs(phi_atp_val - self.k_rest * c_atp_) < 1e-5

        if self.gamma_1 > 0:
            # Use a constant gamma value

            # Determine rest phase via negative ATP drive or near-rest ATP usage.
            cond = np.logical_or(phi_atp_val < 0.001, np.abs(phi_atp_val - self.k_rest * c_atp_) < 1e-6)
            
            # If cond is scalar use scalar gamma, otherwise build elementwise gamma array
            if np.isscalar(cond):
                gamma_val = self.gamma_1 if cond else 1.0
            else:
                gamma_val = np.where(cond, self.gamma_1, 1.0)
        
        else: 
            # Use a sigmoidal gamm value
            gamma_val = 1 / (1 + np.exp(2 * (phi_atp_val - self.k_rest * c_atp_ - 1.5)))

        return gamma_val
    
    def phi_oxphos(self, t, c_adp_, c_atp_):
        # Add a scaling factor to account for different rates during recovery and rest

        # Compute gamma
        gamma = self.gamma_fun(t, c_adp_, c_atp_)


        c_adp_safe = np.maximum(c_adp_, 0.0)
        ratio = np.maximum(c_adp_safe / self.K_adp, 0.0)
        ratio_pow = ratio ** self.nh
        return self.V_max_oxphos * gamma * ratio_pow / (1 + ratio_pow)

    def phi_ck(self, c_atp, c_pcr): 
        c_adp = self.c_a_tot - c_atp 
        c_cr = self.c_c_tot - c_pcr
        return self.V_ck_f / self.K_b / self.K_ia * (c_adp * c_pcr - c_cr * c_atp / self.K_eq)/ \
                    (1 + c_adp / self.K_ia + c_atp / self.K_iq + c_pcr / self.K_ib \
                     + c_adp * c_pcr / self.K_b / self.K_ia + c_cr * c_atp / self.K_iq / self.K_p)
         

    def phi_atp(self, t, c_atp): 
        ''' 
        Function defining ATP use during the contraction 
        
        Modified to simulate the contractions from Barclay et al. 1995
        '''
        return (np.interp(t, self.t_vec, self.e_initial) / 60e-3) + self.k_rest * c_atp  # umol/g/s

    def atp_rhs(self, t, y): 
        atp_curr = np.clip(y[0,], self._eps, self.c_a_tot - self._eps)
        adp_curr = np.maximum(self.c_a_tot - atp_curr, 0.0)
        pcr_curr = np.clip(y[1,], self._eps, self.c_c_tot - self._eps)
        return -self.phi_atp(t, atp_curr) + self.phi_oxphos(t, adp_curr, atp_curr) + self.phi_ck(atp_curr, pcr_curr)
    
    def pcr_rhs(self, t, y): 
        atp_curr = np.clip(y[0,], self._eps, self.c_a_tot - self._eps)
        pcr_curr = np.clip(y[1,], self._eps, self.c_c_tot - self._eps)
        return - self.phi_ck(atp_curr, pcr_curr)
    
    def rhs(self, t, y): 
        return (self.atp_rhs(t,y), self.pcr_rhs(t,y))
    
    def computePi(self): 
        # TODO: implement
        return 0
    
    def solveBioenergetics(self, t_span, c_atp_0, t_eval, e_initial):
        # print('Solving bioenergetics...')
        # print(f'K_adp = {self.K_adp}, V_max_oxphos = {self.V_max_oxphos}')

        self.t_vec = t_eval
        
        # Scale e_initial to be in units of umol
        self.e_initial = e_initial 

        # Calculate the ICs assuming that c_pcr_0 is known
        c_adp_0 = c_atp_0 * (self.c_c_tot - self.c_pcr_0) / (self.K_eq * self.c_pcr_0)
        if self.gamma_1 == 0:
            gamma_0 = 1
        else: 
            gamma_0 = self.gamma_1

        self.k_rest = self.V_max_oxphos * gamma_0 * (c_adp_0 / self.K_adp)**self.nh / c_atp_0 / (1 +(c_adp_0 / self.K_adp)**self.nh)
        # print(f'k_rest = {self.k_rest}')
    
        # Define resting rate 
        self.phi_rest = self.k_rest * c_atp_0

        # Initial conditions 
        # c_pcr_0 calculated as in eqn 11 Vicini 2000 
        c_pcr_0 = (c_atp_0 * self.c_c_tot) / (c_atp_0 + self.K_eq * c_adp_0)
        
        y_0 = (c_atp_0, c_pcr_0)

        self.c_a_tot = c_atp_0 + c_adp_0

        sol = solve_ivp(self.rhs, t_span, y_0, "BDF", max_step = 0.01, t_eval = t_eval)
        
        return sol
    
    def computeRecoveryEnergetics(self, t, c_atp): 

        c_adp_ = np.maximum(self.c_a_tot - c_atp, 0.0)

        gamma_scaler = self.gamma_fun(t, c_adp_, c_atp)

        # Here we assume self.r_rec is in units of J / mol
        # Converted phi_oxphos from umol/s/g to mol/s/g
        energy_rate = self.r_rec * 1e-6  * (self.phi_oxphos(t, c_adp_, c_atp) - gamma_scaler * self.phi_rest)  # subtract resting rate, times 1e6 to account for phi_oxphos in units of umol/s/g  
        # energy_rate = self.r_rec * self.phi_oxphos(c_adp_) * 1e-6 

        return energy_rate # W / g / s

