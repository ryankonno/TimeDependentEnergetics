import numpy as np 
class Bioenergetics(): 
    def __init__(self, k_recpcr, k_m, t_q_i, q_i): 
        # Initialize the main parameters
        self.k_recpcr = k_recpcr
        self.k_m =  k_m 

        # Define the initial heat used during the contraction as a vector
        # NOTE: We assume q_i is in units of J/g/s!!!!!
        self.t_q_i = t_q_i 
        self.q_i = q_i

        # Counter for the cycles 
        self.cycle_count = 1 # NOTE: starts at 1

    # Compute the recovery of ATP 
    def pcrRecoveryRate(self, t, y): 
        '''
        Function of time and current pcr concentration 
        '''
        pcr, Ma = y

        # New model 2a
        dpcrdt = - self.ATP(t) + self.k_recpcr * Ma * (pcr)
        dMadt = self.k_m * (pcr - Ma)

        # Return rate of recovery 
        return (dpcrdt, dMadt)

    def ATP(self, t): 
        ''' 
        Function defining ATP use during the contraction 

        Interpolate given ATP_vec 
        '''
        return np.interp(t, self.t_q_i, self.q_i) / 60e-3 # umol/g/s
    
    def solveODE(self, t_start, t_end, t_vec = None):
        '''
        Function to solve the ode 

        t_vec is a vector of times at which to compute the solution
        '''
        # Initial conditions for ode
        PCr_initial = 0.0  # Initial d[PCr] 
        activation_initial = 0.0  # Initial mitochondrial activation
        y0 = [PCr_initial, activation_initial]

        # Solve the ode 
        from scipy.integrate import odeint, solve_ivp
        tspan = (t_start, t_end)

        # Solve the IVP 
        sol = solve_ivp(self.pcrRecoveryRate, tspan, y0, max_step = 0.001, t_eval=t_vec)

        return sol 
    
    def computeEnergetics(self, pcr, Ma, r_r): 
        '''
        Function to compute the energetic rate of the recovery process 
        '''
        return r_r * self.k_recpcr * Ma * pcr