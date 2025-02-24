'''
Optimizer class to run the optimization given an ideal force trace 

Implemented following the methods by Murtola and Richards 2024

Ryan Konno
'''
import numpy as np

class ContractionOptimizer(): 
    def __init__(self, full_mu_array, tau_d_array_full, size_array_full, params): 

        self.full_mu_array = full_mu_array
        self.tau_d_array_full = tau_d_array_full
        self.size_array_full = size_array_full

        # Counter for optimization
        self.Niter = 0

        self.params = params
    
    def computeExcitation(self, t_vec, force_exp, a1, b, td): 

        # Define the time delay in terms of the index 
        idx_shift = np.floor(td * self.params['fsamp']).astype(int)

        # Define the current (shifted)
        safe_base = np.maximum(force_exp[idx_shift:], 0)

        # Perform the power operation
        exc_ = a1 * safe_base**b
        # exc_ = a1 * (force_exp[idx_shift:] / 1)**b

        # pad the current with zeros to maintain the same length as the input 
        padlength = np.size(force_exp) - np.size(exc_)
        exc_pad = np.pad(exc_, (0,padlength))

        # fig,ax = plt.subplots(layout = 'constrained',figsize = (6,4)) 
        # ax.plot(t_vec, exc_pad)
        # ax.set_xlabel('Time (s)')
        # ax.set_ylabel('current_pad (normalized)')
        # plt.savefig('Figures/' + run_id + '/Pred_Current.pdf')
        # plt.close()

        # Interpolate to remove nan values
        valid = ~np.isnan(exc_pad) # Indices where values are not NaN
        indices = np.arange(len(exc_pad))
        exc_pad = np.interp(indices, indices[valid], exc_pad[valid])# Linear interpolation

        # Add t_vec to the object (may be used later) 
        self.t_vec = t_vec

        return exc_pad
    
    def computeForce(self, t_vec, excitation_vec, output_all = False): 
        '''
        Function to compute the force given an excitation
        '''
        #_________________________________________________
        # Create the predictive model simulator 
        from lib.ContractionPredictor import ContractionPredictor

        self.params['suppress_output'] = True # Suppress plotting updates
        contr_predictor = ContractionPredictor(self.params) 

        # Define the excitation to supply the contraction predictor 
        contr_predictor.defineExcitation(t_vec, excitation_vec)

        #_________________________________________________
        '''
        Simulate the contraction
        '''
        #Simulate discharges
        dt_mat = contr_predictor.simulateMUPool(self.full_mu_array, self.tau_d_array_full, self.size_array_full)

        # Simulate activation
        act, act_mat, act_mat_scaled = contr_predictor.simulateExcAct()

        # Simulate mechanics
        t_vec_mech, e_m_, force_m = contr_predictor.simulateMech()
        
        if output_all:
            # If we want to output all the contraction data then package the output
            #       NOTE: This should not be called during optimization!
            e_m = np.interp(self.t_vec, t_vec_mech, e_m_)
            dedt_m = np.interp(self.t_vec, t_vec_mech[0:-1], np.diff(e_m) / np.diff(t_vec))
            force_m = np.interp(self.t_vec, t_vec_mech, force_m)

            # Also compute the energetics 
            energy_dict = contr_predictor.simulateEnergetics()

            # Create the output
            output = {
                'dt_mat': dt_mat, 
                'act': act, 
                'act_mat': act_mat, 
                'act_mat_scaled': act_mat_scaled, 
                't_vec': t_vec, 
                'e_m': e_m, 
                'dedt_m': dedt_m,
                'force_m': force_m,
                'energy_dict': energy_dict
            } 
        else: # otherwise, we just return the force
            output = force_m

        return output
    
    def obj_fun(self, x): 

        # Get the current force values
        force_new = self.computeForce(self.t_vec, self.computeExcitation(self.t_vec, self.force_ideal_vec, *x))

        # Lets assume an L2 norm for now
        error = np.linalg.norm(self.force_ideal_vec - force_new) / np.linalg.norm(self.force_ideal_vec)

        # Minimize the difference between the integrals of the two curves 
        # int_current_force = np.sum(current_force[0:-1] * np.diff(time))
        # int_ideal_force = np.sum(force_ideal[0:-1] * np.diff(time))
        # error = np.abs(int_ideal_force  - int_current_force)/np.abs(int_ideal_force)

        # Minimize difference between the integrals and the maximum force values
        # TODO: try this one
        # error = np.abs(int_ideal_force  - int_current_force)/np.abs(int_ideal_force) + np.abs(max(current_force) - max(force_ideal)) / np.abs(max(force_ideal))

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(self.t_vec, self.computeExcitation(self.t_vec, self.force_ideal_vec, *x) )
        # plt.plot(self.t_vec, force_new)
        # plt.show()

        return error
    
    def runOptimization(self, t_vec, force_ideal_vec): 
        '''
        Function to run the optimization of the force vector
        '''

        # Defien key variables
        self.t_vec = t_vec
        self.force_ideal_vec = force_ideal_vec

        # x0 = (0.6,    0.7,    0.2) # tramp = 0.8s # Modified for specific trace (exp scale )
        x0 = (0.97,    1.02,    0.12)  # tramp = 0.4s  # Modified for specific trace (lin scale)
        # x0 = (0.95  ,  1.15 ,   0.2) # tramp = 1.6 # ***Standard option***

        # Run the optimziation
        from scipy.optimize import minimize
        print('Running optimization...')
        print('___________________________________________________________________')
        print('{0:4s},   {1:9s},   {2:9s},   {3:9s},   {4:9s}'.format('Iter', 'a1', 'b', 'td', 'fval'))


        # Nelder-Mead optimization
        results = minimize(self.obj_fun, x0, callback = self.callbackFun, method = 'Nelder-Mead', \
                           bounds=((0, 1000), (0.0001, 10), (0.001, 0.5)), options = {'maxiter': self.params['max_iter'], 'disp': True})
        

        # Initialize iteration counter
        
        # Dual-annealing optimization
        # print('note: using dual annealing optimization')
        # self.Niter = 0
        # from scipy.optimize import dual_annealing
        # results = dual_annealing(
        #     self.obj_fun,
        #     bounds=((0, 10), (0.0001, 10), (0.001, 0.5)),
        #     callback=self.callbackFun_Annealing,
        #     maxiter=self.params['max_iter'],  # Replace with your defined `max_iter`
        # )
        
        # Return the optimized value for the excitation
        return results.x

    # Define the callback function
    def callbackFun_Annealing(self, x, f, context):
        """
        Callback function for dual_annealing.
        Called after each iteration or when a special context occurs.
        """
        print('{0:4d},   {1: 3.6f},   {2: 3.6f},   {3: 3.6f},   {4: 3.6f}'.format(self.Niter, x[0], x[1], x[2], f))
        self.Niter += 1

    # Define the callback function
    def callbackFun(self, xk): 
        fun = self.obj_fun(xk)

        # print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}'.format(Niter, xk[0], xk[1], xk[2], xk[3], fun))
        print('{0:4d},   {1: 3.6f},   {2: 3.6f},   {3: 3.6f},   {4: 3.6f}'.format(self.Niter, xk[0], xk[1], xk[2], fun))
        self.Niter += 1