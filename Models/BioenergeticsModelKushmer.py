import numpy as np 
from scipy.integrate import solve_ivp
class Bioenergetics(): 
    def __init__(self, params, t_q_i, q_i, muscle = 'SOL'): 
        
        # Define the muscle 
        self.muscle = muscle 

        # Define the initial heat (used to drive the model) 
        self.t_q_i = t_q_i 
        self.q_i = q_i 

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
         
    def phi_atp(self, t, c_atp): 
        ''' 
        Function defining ATP use during the contraction 
        '''

        return np.interp(t, self.t_q_i, self.q_i) / 60e-3 # umol/g/s

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

        sol = solve_ivp(self.rhs, t_span, y_0, "BDF", max_step = 0.01, t_eval=self.t_q_i)
        
        return sol
    
    def computeRecoveryEnergetics(self, c_atp): 

        c_adp_ = self.c_a_tot - c_atp 

        energy_rate = self.r_rec * self.phi_oxphos(c_adp_)

        return energy_rate 
