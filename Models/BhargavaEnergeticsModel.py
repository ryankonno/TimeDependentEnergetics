'''
Implementation of the Bhargava et al., 2004 model 

Ryan Konno 
r.konno@uq.edu.au
The University of Queensland
'''
# Import 
import numpy as np

class BhargavaEnergetics():
    def __init__(self, alpha_f = params["alpha_f"], recruitment = "Orderly"):
        # Import any important energetic related parameters here
        self.tstim = 0 # Time muscle has been excited above 10\%
        self.tau_phi = params["tau_phi"]

        self.m = params["m"]
        self.alpha_f = alpha_f
        self.alpha_s = 1-self.alpha_f # Chosen to maintain a total muscle volume of 1
        self.q_a_f = params["q_a_f"]
        self.q_a_s =params["q_a_s"]
        self.q_m_f = params["q_m_f"]
        self.q_m_s = params["q_m_s"]

        # Parameter to set the recruitment type
        self.recruitment = recruitment
        if self.recruitment == "Reversed":
            print("WARNING: Using a reversed recruitment pattern!!")

        # Initialize activation model
        self.act_model = ActivationClass()
        self.hill_model = ZajacHillModel()

        # Solve the activation model (does not depend on the mechanics or heat)
        print("Solving the activation model")
        self.act_tvals,self.actvals,self.uvals = self.act_model.solveActivation()

        # Solve the mechanical model
        print("Solving the mechanical model")
        self.F_tvals, self.Fvals = self.hill_model.SolveForF()



    ############################################################################
    # Heat Rates
    # TODO: !!! Need to go through and check units (needs to compare with the experimntal constants)
    # Activation heat rate
    def dQadt(self,t,u):
        return self.decay(t,u) * self.m * (self.alpha_f * self.q_a_f * self.u_fast(u) \
                + self.alpha_s * self.q_a_s * self.u_slow(u))

    # Maintenance heat rate
    def dQmdt(self, u, e_ce):
        return self.L(e_ce) * self.m * (self.alpha_f * self.q_m_f * self.u_fast(u) \
                + self.alpha_s * self.q_m_s * self.u_slow(u))

    # Shortening/lengthening heat rate
    def dQsldt(self, e_ce, dedt_ce, a=1):
        # !!! Note that the stretches used to calculate some of the forces should be dependent on the whole muscle stretch not just CE stretch
        F_iso = self.hill_model.forceIsoNormalized(e_ce,a)
        F_hill = self.hill_model.hillModelCalculationNormalized(e_ce,dedt_ce,a)

        const_short = (0.16 * F_iso+ 0.18 * F_hill)
        const_long = 0.157 * F_hill

        const =  const_short * (dedt_ce <= 0) + const_long * (dedt_ce > 0)

        # Equation (8) Bhargava et al. 2004
        return - const * dedt_ce

    # Basil metabolic rate
    def dQbdt(self):
        return 0.0225 * self.m

    # Internal work rate
    def dWdt(self, t, dedt_ce):
        return np.interp(t,self.F_tvals,self.Fvals) * dedt_ce

    # Total energy rate
    def dEdt(self, t, E):
        # Need this to be a function of t and E
        # define u, e_ce, dedt_ce, a
        u_curr = np.interp(t,self.act_tvals,self.uvals) # Interpolate to get the excitation
        F = np.interp(t,self.F_tvals,self.Fvals) # Interpolate force values from Hill solution
        e_ce_curr = self.hill_model.get_e_m(t,F)
        # e_ce_curr = self.hill_model.e_mtu(t)

        # Evaluate the current stretch rate of the CE
        dedt_ce_curr = self.hill_model.dedt_m(t,F)

        # Get the current activation
        a_curr = np.interp(t,self.act_tvals,self.actvals)

        return self.dQadt(t,u_curr) + self.dQmdt(u_curr, e_ce_curr) + self.dQsldt(e_ce_curr, dedt_ce_curr, a_curr)\
                + self.dQbdt() + self.dWdt(t, dedt_ce_curr)

    ############################################################################
    # Additional function
    def decay(self,t,u):
        # return 0.06 + np.exp(-self.tstim * u / self.tau_phi)
        # print(np.size(u))
        if np.size(u) > 1:
            tstim = self.gettstim(t,u)
            return 0.06 + np.exp(-(t - tstim) * ((t-tstim)>0) * u / self.tau_phi)
        else:
            # Condition for the ode solvers
            # print("Solving ode")
            if self.tstim == 0:
                self.tstim = self.gettstim(t,u)
                # print(0.06 + np.exp(-self.tstim * u / self.tau_phi))
                return 0.06 + np.exp(-self.tstim * u / self.tau_phi)
            else:
                # print(0.06 + np.exp(-(t - self.tstim) * u / self.tau_phi))
                return 0.06 + np.exp(-(t - self.tstim) * u / self.tau_phi)

    def gettstim(self,t,u):
        # Get the time from u>0.1
        # Currently assuming that this happens only once
        tstim = 0
        if np.size(u) > 1:
            for i, val in enumerate(u):
                if val > 0.1:
                    tstim = t[i]
                    break
            return tstim
        else:
            # Condition for the ode solvers
            if u > 0.1:
                return t
            else:
                return 0
            # return (t - tstim) * ((t-tstim)>0)

    def u_fast(self,u):
        if self.recruitment == "Reversed":
            return self.act_model.u_slow(u)
        else:
            return self.act_model.u_fast(u)


    def u_slow(self,u):
        if self.recruitment == "Reversed":
            return self.act_model.u_fast(u)
        else:
            return self.act_model.u_slow(u)

    # Stretch dependence of maintenance heat rate
    def L(self, e):
        return 0.5 * (e <= 0.5) + e * (e>0.5)*(e<=1) + (-2*e + 3)*(e>=1)*(e<1.5)

    ############################################################################
    # Functions to solve the energy rate equation
    # Solve for the whole energy
    def SolveForE(self):
        print("Solving the heat model")
        tspan = (0,params["tend"])
        E0 = (0,)
        return solve_ivp(self.dEdt,tspan,E0,max_step=params["dt_max"])

    # Get the activation heat
    def SolveForQa(self):
        tspan = (0,params["tend"])
        Q0 = (0,)
        def rhs(t,Q):
            u = np.interp(t,self.act_tvals,self.uvals)
            return self.dQadt(t,u)
        return solve_ivp(rhs,tspan,Q0,max_step=params["dt_max"])

    # Get the maintenance heat
    def SolveForQm(self):
        tspan = (0,params["tend"])
        Q0 = (0,)
        def rhs(t,Q):
            u = np.interp(t,self.act_tvals,self.uvals)
            # e = self.hill_model.e_mtu(t)
            F = np.interp(t,self.F_tvals,self.Fvals)
            e = self.hill_model.get_e_m(t,F)
            return self.dQmdt(u,e)
        return solve_ivp(rhs,tspan,Q0,max_step=params["dt_max"])

    # Get the shortening/lengthening heat
    def SolveForQsl(self):
        tspan = (0,params["tend"])
        Q0 = (0,)
        def rhs(t,Q):
            u = np.interp(t,self.act_tvals,self.uvals)
            # e = self.hill_model.e_mtu(t)
            F = np.interp(t,self.F_tvals,self.Fvals)
            e = self.hill_model.get_e_m(t,F)
            dedt = self.hill_model.activeFVInverse(F)
            act = np.interp(t,self.act_tvals,self.actvals)
            return self.dQsldt(e,dedt,act)
        return solve_ivp(rhs,tspan,Q0,max_step=params["dt_max"])

    # Get the work
    def SolveForW(self):
        tspan = (0,params["tend"])
        Q0 = (0,)
        def rhs(t,Q):
            F = np.interp(t,self.F_tvals,self.Fvals)
            dedt = self.hill_model.activeFVInverse(F)
            return self.dWdt(t,dedt)
        return solve_ivp(rhs,tspan,Q0,max_step=params["dt_max"])
