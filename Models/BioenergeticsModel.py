'''
Bioenergetics model code. 

Based on the following models 

Kushmerick, M. J. (1998). Energy balance in muscle activity: Simulations of ATPase coupled to oxidative phosphorylation and to creatine kinase. Comparative Biochemistry and Physiology Part B: Biochemistry and Molecular Biology 120, 109–123.

Vicini, P. and Kushmerick, M. J. (2000). Cellular energetics analysis by a mathematical model of energy balance: estimation of parameters in human skeletal muscle. American Journal of Physiology-Cell Physiology 279, C213–C224.

Ryan Konno
r.konno@uq.edu.au
The University of Queensland 
'''

###################################################################
# Import
import numpy as np
from scipy.integrate import solve_ivp

###################################################################
# Implementation of the bioenergetics model 
class Bioenergetics(): 
    def __init__(self, params): 
        
        # Define the muscle 
        self.muscle = params['muscle'] 

        # Initialize initial and total concentrations
        self.c_c_tot = params[self.muscle]['c_c_tot'] # mM
        self.c_pcr_0 = params[self.muscle]['c_pcr_0'] # mM

        # Initialise maximum rates 
        self.V_max_oxphos = params[self.muscle]['V_max_oxphos'] # mM/s

        # Gamma parameter to account for different rest-recovery rates
        self.gamma_1 = params[self.muscle]['gamma']
        
        # Parameters governing time course (optimised)
        self.K_adp = params[self.muscle]['K_adp']
        self.nh = params[self.muscle]['nh']

        # Other parameters fixed from Kushmerick 1998
        self.V_ck_f =  params['V_ck_f'] # mM/s
        self.K_b = params['K_b']  # mM
        self.K_ia = params['K_ia'] # mM
        self.K_eq = params['K_eq']
        self.K_iq = params['K_iq'] # mM
        self.K_ib = params['K_ib'] # mM
        self.K_p = params['K_p'] # mM

        # Energetic rate parameter 
        self.r_rec = params[self.muscle]['r_rec'] # J/mol

        # Small tolerance used to keep concentrations in a numerically safe range.
        self._eps = 1e-12

    ###################################################################
    def c_adp(self, c_atp_):
        '''
        Calculate concentration of ADP
        '''
        return self.c_a_tot - c_atp_ # mM
    ###################################################################
    def c_cr(self, c_pcr_): 
        '''
        Calculate Cr concentration
        '''
        return self.c_c_tot - c_pcr_ # mM
    ###################################################################    
    def gamma_fun(self, t, c_adp_, c_atp_):
        '''
        Compute the gamma value
        '''
        # Evaluate phi_atp (may be scalar or array) and form a boolean mask
        phi_atp_val = self.phi_atp(t, c_atp_)

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
            # Use a sigmoidal gamma value
            gamma_val = 1 / (1 + np.exp(2 * (phi_atp_val - self.k_rest * c_atp_ - 1.5)))

        return gamma_val
    ###################################################################
    def phi_oxphos(self, t, c_adp_, c_atp_):
        '''
        Compute the oxidative phosphorylation flux 
        '''
        # Compute gamma
        gamma = self.gamma_fun(t, c_adp_, c_atp_)

        # Compute c_adp, assume a minimum of 0 
        c_adp_safe = np.maximum(c_adp_, 0.0)
        ratio = np.maximum(c_adp_safe / self.K_adp, 0.0)
        ratio_pow = ratio ** self.nh
        return self.V_max_oxphos * gamma * ratio_pow / (1 + ratio_pow) # umol/g/s
    ###################################################################
    def phi_ck(self, c_atp, c_pcr): 
        '''
        Compute the creatine-kinase flux
        '''
        c_adp = self.c_a_tot - c_atp 
        c_cr = self.c_c_tot - c_pcr
        return self.V_ck_f / self.K_b / self.K_ia * (c_adp * c_pcr - c_cr * c_atp / self.K_eq)/ \
                    (1 + c_adp / self.K_ia + c_atp / self.K_iq + c_pcr / self.K_ib \
                     + c_adp * c_pcr / self.K_b / self.K_ia + c_cr * c_atp / self.K_iq / self.K_p) # umol/g/s
         
    ###################################################################
    def phi_atp(self, t, c_atp): 
        ''' 
        Function computing the ATP flux
        '''
        return (np.interp(t, self.t_vec, self.e_initial) / 60e-3) + self.k_rest * c_atp  # umol/g/s
    ###################################################################
    def atp_rhs(self, t, y): 
        '''
        ATP equations right-hand side
        '''
        atp_curr = np.clip(y[0,], self._eps, self.c_a_tot - self._eps)
        adp_curr = np.maximum(self.c_a_tot - atp_curr, 0.0)
        pcr_curr = np.clip(y[1,], self._eps, self.c_c_tot - self._eps)
        return -self.phi_atp(t, atp_curr) + self.phi_oxphos(t, adp_curr, atp_curr) + self.phi_ck(atp_curr, pcr_curr) # umol/g/s
    ###################################################################
    def pcr_rhs(self, t, y): 
        '''
        PCr equations right-hand side
        '''
        atp_curr = np.clip(y[0,], self._eps, self.c_a_tot - self._eps)
        pcr_curr = np.clip(y[1,], self._eps, self.c_c_tot - self._eps)
        return - self.phi_ck(atp_curr, pcr_curr) # umol/g/s
    ###################################################################
    def rhs(self, t, y): 
        '''
        Right-hand side of the ODE system
        '''
        return (self.atp_rhs(t,y), self.pcr_rhs(t,y))
    ###################################################################
    def solveBioenergetics(self, t_span, c_atp_0, t_eval, e_initial):
        '''
        Solve the bioenergetics model 

        Input: 
            t_span: Range of time values 
            c_atp_0: Initial concentration of ATP 
            t_eval: Time values at which to evaluate the model 
            e_initial: Energy consumed by the initial processes 
        Output: 
            sol: Solution to solve_ivp. Gives the concentration of ATP and PCr.
        '''
        # Set vector on which to evaluate the model 
        self.t_vec = t_eval

        # Define the initial energy used
        self.e_initial = e_initial 

        # Calculate the initial conditions assuming that c_pcr_0 is known
        c_adp_0 = c_atp_0 * (self.c_c_tot - self.c_pcr_0) / (self.K_eq * self.c_pcr_0)

        # Account for gamma 
        if self.gamma_1 == 0:
            gamma_0 = 1
        else: 
            gamma_0 = self.gamma_1

        # Calculate k_rest accounting for the gamma value 
        self.k_rest = self.V_max_oxphos * gamma_0 * (c_adp_0 / self.K_adp)**self.nh / c_atp_0 / (1 +(c_adp_0 / self.K_adp)**self.nh)
    
        # Define resting rate 
        self.phi_rest = self.k_rest * c_atp_0

        # c_pcr_0 calculated as in eqn 11 Vicini 2000 
        c_pcr_0 = (c_atp_0 * self.c_c_tot) / (c_atp_0 + self.K_eq * c_adp_0)
        
        # Initial conditions 
        y_0 = (c_atp_0, c_pcr_0)

        # Set the total adenosine concentration
        self.c_a_tot = c_atp_0 + c_adp_0

        # Solve the IVP
        sol = solve_ivp(self.rhs, t_span, y_0, "BDF", max_step = 0.01, t_eval = t_eval)
        
        return sol
    ###################################################################
    def computeRecoveryEnergetics(self, t, c_atp): 
        '''
        Compute the recovery energetic rate 

        Input: 
            t: time vector 
            c_atp: concentration of ATP at time points
        Output:
            energy_rate: Energetic rate in W/g
        '''

        c_adp_ = np.maximum(self.c_a_tot - c_atp, 0.0)

        # Compute the gamma value
        gamma_scaler = self.gamma_fun(t, c_adp_, c_atp)

        # Here self.r_rec is in units of J / mol
        # Converted phi_oxphos from umol/s/g to mol/s/g
        energy_rate = self.r_rec * 1e-6  * (self.phi_oxphos(t, c_adp_, c_atp) - gamma_scaler * self.phi_rest)

        return energy_rate # W / g 

