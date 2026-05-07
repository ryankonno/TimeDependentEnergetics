'''
Mechanical model code. 

Model equations are based on Dick et al. 2017.

Note this version of the model does not capture a series elastic element. 

Ryan Konno
r.konno@uq.edu.au
The University of Queensland 
'''
################################################################################
# Import
import numpy as np
from scipy.integrate import solve_ivp

################################################################################
class MechModel():
    def __init__(self, l_0, dedt_ce_max, kappa, k_see):
        # Optimal fibre length
        self.l_0 = l_0                    
        # Maximum shortening strain rate
        self.dedt_ce_max = dedt_ce_max
        # Curvature of force-velocity relationship
        self.kappa = kappa
        # Series elastic component
        self.k_see = k_see

        return None

    ################################################################################
    '''
    Intrinsic muscle properties 
    Define force-length and force-velocity properties for the musclel. From Dick et al. 2017
    '''
    def F_la(self, e_ce):
        '''
        Force-strain relationship
        '''
        # parameters
        skew = 0.6
        round = 2.3
        width = 0.3

        # Clip e_ce to avoid negative values and numerical issues
        e_ce_safe = np.clip(e_ce, 1e-6, None)
        result = np.exp(-np.abs((e_ce_safe**skew -1)/width)**round)
        return np.clip(result, 0, 1)

    def F_va(self, dedt_ce):
        '''
        Force-strain rate relationship
        '''
        return (1+dedt_ce/self.dedt_ce_max)/(1-dedt_ce/self.dedt_ce_max/self.kappa) * (dedt_ce < 0) \
                + (1.5 - 0.5*(1-dedt_ce/self.dedt_ce_max)/(1+7.56 * dedt_ce/ self.dedt_ce_max / self.kappa))*(dedt_ce >0)\
                + (dedt_ce == 0)
    
    def F_va_inverse(self, F):
        '''
        Inverse of the force-velocity relationship
        '''
        return (self.dedt_ce_max * (self.kappa*(F-1)/(F+self.kappa)) * (F <= 1)\
                + (2 * (-1 + F)) / (3 * 7.56 / self.kappa + 1 - 2 * 7.56 * F / self.kappa) * (F > 1))
    
    def F_lp(self, e_ce): 
        '''
        Passive force-length relationship
        '''
        return (2.64 * e_ce**2 - 5.3 * e_ce + 2.66) * (e_ce > 1)
    ################################################################################
    #           Function to compute the muscle force 
    def computeForce(self, act_, e_ce_, dedt_ce_): 
        '''
        Function to compute the scaled muscle force give act, e_m, dedt_m
        '''
        return act_ * self.F_la(e_ce_) * self.F_va(dedt_ce_) + self.F_lp(e_ce_) # Unitless
    

    # ################################################################################
    # #       Implement a model with a damper to deal with sudden length changes
    # def rhs(self, t, x): 
    #     dx = x[1]
    #     dv = x[0] 
        

    #     beta = self.damp_coeff 
    #     k = self.stiff_coeff
        
    #     # Interpolate to get current values at this time step
    #     act_curr = np.interp(t, self.t_vec, self.act)
    #     e_tot_curr = np.interp(t, self.t_vec, self.e_tot)
    #     dedt_tot_curr = np.interp(t, self.t_vec, self.dedt_tot)

    #     # Print current state 
    #     # print(f't = {t}, dx = {dx}, dv = {dv}, a = {act_curr}')
        
        
    #     # Compute CE strain and strain rates 
    #     e_ce_curr = (e_tot_curr * self.l_tot_0 - (dx * self.l_see_0) ) / self.l_ce_0
    #     # dedt_ce_curr = (e_ce_curr - self.e_ce_prev) / (t - self.t_prev) # Maybe we can compute this with prescibed length change .... next line
    #     dedt_ce_curr = (dedt_tot_curr * self.l_tot_0 - (dv * self.l_see_0) ) / self.l_ce_0 #

    #     # print(f'    e_ce_curr = {e_ce_curr}, dedt_ce_curr = {dedt_ce_curr}')
        
    #     # # Update previous value 
    #     # self.e_ce_prev = e_ce_curr
    #     # self.dedt_ce_prev = dedt_ce_curr
        
    #     # Compute v_see rhs 
    #     # Use dx (position) for spring term
    #     dv_see_rhs = (- beta * dv - k * dx + self.computeForce(act_curr, e_ce_curr, dedt_ce_curr)) / self.mass
        
    #     # Compute x_see rhs 
    #     dx_see_rhs = dv
        

    #     return  (dv_see_rhs, dx_see_rhs)
    
    # ################################################################################
    # def solveModel(self, t_vec,  act, e_tot_, dedt_tot_, l_0_, mass, damp_coeff = 100, stiff_coeff = 1000, l_see_rat = 0.01): 
        
    #     # Initials HO parameters 
    #     self.damp_coeff = damp_coeff 
    #     self.stiff_coeff = stiff_coeff
    #     self.l_see_rat = l_see_rat
    #     self.mass =mass

    #     # Define model initial state (mainly the lengths)
    #     self.l_ce_0 = l_0_ 
    #     self.l_see_0 = self.l_see_rat * self.l_ce_0 
    #     self.l_tot_0 = self.l_ce_0 + self.l_see_0 

    #     # Assume input is stretch and stretch rates
    #     self.t_vec = t_vec
    #     self.act = act 
    #     self.e_tot = e_tot_
    #     self.dedt_tot = dedt_tot_
    #     self.e_ce_prev = 1 # Set previous value to compute rate of change
    #     self.t_prev = 0 # Set previous value to compute rate of change

    #     # ICs 
    #     x0 = (0, 1)
    #     t_span = (t_vec[0], t_vec[-1])

    #     # Solve the model using solve_ivp with stricter tolerances
    #     sol = solve_ivp(self.rhs, t_span, x0, method='BDF', t_eval=t_vec, rtol=1e-6, atol=1e-9, max_step = 0.0001)
    #     e_tot_sol = sol.y[1,]
    #     dedt_tot_sol = sol.y[0,]
        
    #     e_ce_curr = (self.e_tot * self.l_tot_0 - (e_tot_sol * self.l_see_0) ) / self.l_ce_0
    #     dedt_ce_curr = (self.dedt_tot * self.l_tot_0 - (dedt_tot_sol * self.l_see_0) ) / self.l_ce_0 #

    #     # Compute the muscle force given the ce strain and strain rate
    #     force_vec = self.computeForce(self.act, e_ce_curr, dedt_ce_curr)

    #     return force_vec
