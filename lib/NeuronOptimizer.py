'''
Code to optimize motor neuron parameter values

Ryan Konno
'''
import numpy as np
import matplotlib.pyplot as plt
import time

class NeuronOptimizer:
    def __init__(self, params, contr_id = 1): 
        self.params = params
        
        # Define key parameters in converting from the excitation values to the input current
        self.I0 = params['I0']
        self.G = params['G']

        # Define contr id for saving files 
        self.contr_id = contr_id

        # Sample frequency of the experimental data
        self.fsamp_exp = 2048 

    def optimizeNeuronProperties(self, excitation_, r_ideal_smoothed, neuron, mu_id = 1):
        ''' 
        This function is used to optimize over a single MU
        specified via mu_num.
        ''' 

        # Define the objective function function here
        #       This is where the input neuron is used.
        def J(x): 
            # Optimize both values over the entire firing rate trace
            # Update the neuron values 
            neuron.updateTaud(x[0])
            neuron.updateSize(x[1])

            # Solve the LIF model
            t_vec_, V_vec_, t_fire_vec_ = self.solveLIF(neuron,excitation_)

            # Smooth the data 
            from lib.smoothDT import smoothDT
            r_sim_smoothed = smoothDT(t_vec_, t_fire_vec_, 1/self.params['dt'])

            # Plot the current status
            # ax.plot(t_vec_, r_sim_smoothed, label='opt_iter')
            
            # Return the objective function 
            # print(f'Error = {np.sqrt(np.mean((r_sim_smoothed - r_ideal_smoothed)**2))/ np.sqrt(np.mean((r_ideal_smoothed)**2))}')
            return np.sqrt(np.mean((r_sim_smoothed - r_ideal_smoothed)**2))/ np.sqrt(np.mean((r_ideal_smoothed)**2))        

        # Now lets generate the MN that we will optimize
        from scipy.optimize import minimize_scalar, minimize, dual_annealing

        # Define optimization bounds 
        size_min = self.params['size_min']
        size_max = self.params['size_max']
        tau_d_min = self.params['tau_d_min']
        tau_d_max = self.params['tau_d_max']


        # Now run the optimization 
        # print(' Running the optimization...')
        x_0 = (0.2,1e-7) # Define the initial point
        opt_res = minimize(J, x_0, bounds = ((tau_d_min,tau_d_max), (size_min,size_max)), method='Nelder-Mead', tol=1e-7)
        # opt_res = dual_annealing(J, maxiter = 10, bounds = ((tau_d_min,tau_d_max), (size_min,size_max)))


        # Print output of optimization
        # print(' ________________________________________')
        # print(f'Ideal MU size: {size_ideal}')
        # print(f'Ideal tau_d: {tau_d_ideal}')
        # print(f'    Optimized tau_d: {opt_res.x[0]}')
        # print(f'    Optimized MU size: {opt_res.x[1]}')
        # print(f'    Optimized objective: {opt_res.fun}')
        print(f'Optimized MU {mu_id}, tau_d = {opt_res.x[0]}, size = {opt_res.x[1]}')


        # TESTING: Re run the the model with optimal neuron
        neuron.updateTaud(opt_res.x[0])
        neuron.updateSize(opt_res.x[1])
        t_vec_opt, V_vec_opt, t_fire_vec_opt = self.solveLIF(neuron, excitation_)
        fig,ax = plt.subplots(figsize = (6,4), layout = 'constrained')
        
        ax2 = ax.twinx()
        ax2.plot(t_vec_opt, excitation_[np.array(t_vec_opt * self.fsamp_exp, dtype='int')], color='k',linewidth = 0.5, ls = 'dashed')
        ax2.set_ylabel('Excitation (Normalized)')

        ax.plot(t_vec_opt, r_ideal_smoothed, label='Ideal')
        ax.legend()
        from lib.smoothDT import smoothDT
        ax.plot(t_vec_opt, smoothDT(t_vec_opt, t_fire_vec_opt, 1/self.params['dt']))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing rate (Hz)')
        plt.savefig('Figures/' + self.params['run_id'] + '/LIFMN_Fit_Contr' + str(self.contr_id) + '_MU' + str(mu_id) + '.pdf')
        plt.close()

        

        # plt.show()

        return opt_res.x, opt_res.fun, neuron
    
    def optimizeNeuronPropertiesSplit(self, excitation_, r_ideal_smoothed, neuron, mu_id = 1):
        ''' 
        This function is used to optimize over a single MU

        There are a couple differences between this optimization function and the default, basic 
        optimization optimizeNeuronProperties. 

        This function will first optimize the MU size to obtain the correct recruitment threshold, 
        and subsequently optimizes tau_d to over the full firing trace
        ''' 

        # Get the indexes to optimize over 
        idx_ramp_end = 0 # Initialize value to 0
        r_thresh = max(r_ideal_smoothed) # Get first value at 75\% of max
        for idx, val in enumerate(r_ideal_smoothed): 
            if val >= r_thresh:
                idx_ramp_end = idx
                # print(f'idx_ramp_end = {idx_ramp_end}')
                break
        # Get the cropped smoothed ideal discharge rates 
        r_ideal_smoothed_cropped = r_ideal_smoothed[0:idx_ramp_end]

        # Define the objective functions function here
        #       This is where the input neuron is used.
        def J_tau_d(x): 
            # Update the neuron value
            neuron.updateTaud(x)

            # Solve the LIF model
            t_vec_, V_vec_, t_fire_vec_ = self.solveLIF(neuron,excitation_)

            # Smooth the data 
            from lib.smoothDT import smoothDT
            r_sim_smoothed = smoothDT(t_vec_, t_fire_vec_, 1/self.params['dt'])

            # Plot the current status
            # ax.plot(t_vec_, r_sim_smoothed, label='opt_iter') 

            # Compute the error 
            error_r = np.sqrt(np.mean((r_sim_smoothed - r_ideal_smoothed)**2))/ np.sqrt(np.mean((r_ideal_smoothed)**2))
            # print(f'Error = {error_r}')
            
            # Return the objective function 
            return error_r
        
        def J_size(x): 
            # This is a new optimization function to have MU size focus on the RT and tau_d optimize the firing rates

            # Update the neuron values
            neuron.updateSize(x)

            # Print neuron update 
            # print(f'R = {neuron.R_0 }')

            # Solve the LIF model
            t_vec_, V_vec_, t_fire_vec_ = self.solveLIF(neuron,excitation_) 

            # Smooth the data 
            from lib.smoothDT import smoothDT
            r_sim_smoothed = smoothDT(t_vec_, t_fire_vec_, 1/self.params['dt'])

            # We want to optimize over the ramp time
            #   for the current implementation we optimize over the the ramp up, so
            #   we can say we optimize from t \in [0, t_x] 
            #   where t_75 is the first time instant, when we get within x% of the max
            #   this is calculated prior to this function 
            r_sim_smoothed_cropped = r_sim_smoothed[0:idx_ramp_end]


            # Compute the ideal recruitment time given the MU firing rates
            target = 0.01
            idx_ideal_rt = np.abs(r_ideal_smoothed - target).argmin()
            # Compute the simulated recruitment time
            idx_sim_rt = np.abs(r_sim_smoothed - target).argmin()

            # Compute the error 
            error_r = np.sqrt(np.mean((r_sim_smoothed - r_ideal_smoothed)**2))/ np.sqrt(np.mean((r_ideal_smoothed)**2))
            # error_r = np.sqrt(np.mean((r_sim_smoothed_cropped - r_ideal_smoothed_cropped)**2))/ np.sqrt(np.mean((r_ideal_smoothed_cropped)**2))
            
            ###
            # fig, ax = plt.subplots(layout = 'constrained')
            # ax.plot(t_vec_, r_ideal_smoothed, label='Ideal')
            # ax.legend()
            # from lib.smoothDT import smoothDT
            # ax.plot(t_vec_, smoothDT(t_vec_, t_fire_vec_, 1/self.params['dt']))
            # ax2 = ax.twinx()
            # ax2.plot(t_vec_, np.abs(r_sim_smoothed - r_ideal_smoothed), color = 'r')
            # ax.set_xlabel('Time (s)')
            # ax.set_ylabel('Firing rate (Hz)')
            # ax.axvline(t_vec_[idx_ramp_end])
            # ax.set_title(f'size = {x}')
            # plt.show()
            # # time.sleep(1) 
            # plt.close()
            ###

            # print(f'Error = {error_r}')

            return error_r

        

        # Now lets generate the MN that we will optimize
        from scipy.optimize import minimize_scalar, minimize

        # Define optimization bounds 
        size_min = self.params['size_min']
        size_max = self.params['size_max']
        tau_d_min = self.params['tau_d_min']
        tau_d_max = self.params['tau_d_max']

        # Run the optimization over tau_d (scalar optimization)
        opt_res_tau_d = minimize_scalar(J_tau_d, bracket = (tau_d_min, tau_d_max), bounds = (tau_d_min, tau_d_max), method = 'bounded', options={'xatol':1e-8})
        # print(f'Optimized MU {mu_id}, tau_d = {opt_res_tau_d.x}, size = {neuron.size}')

        # Run optimization on the MU size (scalar optimization)
        opt_res_size = minimize_scalar(J_size, bracket = (size_min, size_max), bounds = (size_min, size_max), method = 'bounded', options={'xatol':1e-8})

        # Display output
        print(f'Optimized MU {mu_id}, tau_d = {opt_res_tau_d.x}, size = {opt_res_size.x}')


        # TESTING: Re run the the model with optimal neuron
        neuron.updateTaud(opt_res_tau_d.x)
        neuron.updateSize(opt_res_size.x)

        t_vec_opt, V_vec_opt, t_fire_vec_opt = self.solveLIF(neuron, excitation_)
        fig,ax = plt.subplots(figsize = (6,4), layout = 'constrained')
        
        ax2 = ax.twinx()
        ax2.plot(t_vec_opt, excitation_[np.array(t_vec_opt * self.fsamp_exp, dtype='int')], color='k',linewidth = 0.5, ls = 'dashed')
        ax2.set_ylabel('Excitation (Normalized)')

        ax.plot(t_vec_opt, r_ideal_smoothed, label='Ideal')
        ax.legend()
        from lib.smoothDT import smoothDT
        ax.plot(t_vec_opt, smoothDT(t_vec_opt, t_fire_vec_opt, 1/self.params['dt']))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing rate (Hz)')
        plt.savefig('Figures/' + self.params['run_id'] + '/LIFMN_Fit_Contr' + str(self.contr_id) + '_MU' + str(mu_id) + '.jpg')
        plt.close()

        # Package output
        opt_res = (opt_res_tau_d.x, opt_res_size.x)

        return opt_res, opt_res_tau_d.fun, neuron
    
    def solveLIF(self, neuron_model, excitation):
        '''
        Function to call the MU solver
        '''

        def currentFun(t): 
            '''
            Function to compute the current given some excitation

            The parameters for this function self.G, self.I0 are determined previously through based on the input functions
            '''
            return self.I0 + self.G * excitation[int(t * self.fsamp_exp)]

        t_vec, V_vec, t_fire_vec = neuron_model.solveNeuron(self.params['t_end'], currentFun)
        
        return t_vec, V_vec, t_fire_vec

    def testRunModel(self, excitation_, neuron): 
        '''
        This function is meant to be a test function to make sure the model is properly running 
        '''

        print('Running test model')

        # Define the excitation for the current function
        self.excitation = excitation_

        t_vec, V_vec, t_fire_vec =  self.solveLIF(neuron)

        # Plot the output 
        plt.figure()
        plt.plot(t_vec[t_fire_vec[1:]], 1/(t_vec[t_fire_vec[1:]] - t_vec[t_fire_vec[0:-1]]), ls = 'None', marker = 'o', color = 'k') 
        plt.show()
            
    def optimizeNeuronPropertiesMulti(self, excitation_, r_ideal_smoothed_vec, neuron, Ncontr):
        ''' 
        This function is used to optimize over a single MU
        specified via mu_num.

        Here we can specify a number of contractions to optimize over instead of just a single contraction.

        Assumes excitation has field names ['plateau<N>']['excitation_norm']
        Assumes r_ideal_smoothed has field names ['plateau<N>']
        ''' 

        # Define a vector to store errors fore each contraction 
        error_vec = np.zeros((3,))

        # Define the objective function function here
        #       This is where the input neuron is used.
        def J(x): 

            # Update variables in the parameters 
            self.params['tau_d'] = x[0]
            self.params['size']  = x[1]

            # update the neuron values
            neuron.updateTaud(x[0])
            neuron.updateSize(x[1])

            # Solve the LIF model for each condition 
            for i in range(Ncontr):
                name = 'contr' + str(i)
                self.excitation = excitation_[name]['excitation_norm']
                t_vec_, V_vec_, t_fire_vec_ = self.solveLIF(neuron)

                # Smooth the data 
                from lib.smoothDT import smoothDT
                r_sim_smoothed = smoothDT(t_vec_, t_fire_vec_, 1/self.params['dt'])

                # Define the ideal smoothed vector 
                r_ideal_smoothed = r_ideal_smoothed_vec[name]

                # Compute the error for this contr
                error_vec[i] = np.sqrt(np.mean((r_sim_smoothed - r_ideal_smoothed)**2))/ np.sqrt(np.mean((r_ideal_smoothed)**2))
            
            # Return the sum of the error from each contraction
            return sum(error_vec)
        

        # Now lets generate the MN that we will optimize
        from scipy.optimize import minimize_scalar, minimize

        # Define optimization bounds 
        size_min = self.params['size_min']
        size_max = self.params['size_max']
        tau_d_min = self.params['tau_d_min']
        tau_d_max = self.params['tau_d_max']


        # Now run the optimization 
        print('Running the optimization...')
        x_0 = (0.2,1e-7) # Define the initial point
        opt_res = minimize(J, x_0, bounds = ((tau_d_min,tau_d_max), (size_min,size_max)), method='Nelder-Mead', tol=1e-7, options = {'disp': True})


        # Print output of optimization
        print('________________________________________')
        # print(f'Ideal MU size: {size_ideal}')
        # print(f'Ideal tau_d: {tau_d_ideal}')
        print(f'Optimized tau_d: {opt_res.x[0]}')
        print(f'Optimized MU size: {opt_res.x[1]}')
        print(f'Optimized objective: {opt_res.fun}')


        # TESTING: Re run the the model with optimal neuron
        # neuron.tau_d, neuron.size = opt_res.x[0], opt_res.x[1]
        # t_vec_opt, V_vec_opt, t_fire_vec_opt = self.solveLIF(neuron)
        # fig,ax = plt.subplots(figsize = (6,4), layout = 'constrained')
        
        # ax2 = ax.twinx()
        # ax2.plot(t_vec_opt, self.excitation[np.array(t_vec_opt * self.fsamp_exp, dtype='int')], color='k',linewidth = 0.5, ls = 'dashed')
        # ax2.set_ylabel('Excitation (Normalized)')

        # ax.plot(t_vec_opt, r_ideal_smoothed, label='Ideal')
        # ax.legend()
        # from lib.smoothDT import smoothDT
        # ax.plot(t_vec_opt, smoothDT(t_vec_opt, t_fire_vec_opt, 1/self.params['dt']))
        # ax.set_xlabel('Time (s)')
        # ax.set_ylabel('Firing rate (Hz)')
        # plt.savefig('Figures/LIFMN_Optimization.pdf')

        return opt_res.x, neuron
    
    def optimizeNeuronPropertiesAll(self, excitation, r_ideal_mat, params):
        '''
        This function runs the optimization for all mus identified for a given contraction 

        TODO: Update this code to multithread
        '''
        # Initial vector to store output 
        tau_d_output = np.zeros((len(r_ideal_mat),))
        size_output = np.zeros((len(r_ideal_mat),))

        # Add excitation and r_ideal_mat
        self.excitation = excitation 
        self.r_ideal_mat = r_ideal_mat

        # Call the previous optimization with each neuron 
        # THIS IS A FOR LOOP IMPLEMENTATION... DELETE ONCE MULTITHREADING IS IMPLEMENTED
        # for mu_idx in range(len(r_ideal_mat)): 

        #     print(f'Running optimization on MU {mu_idx}')

        #     # Generate a neuron to use in the optimization 
        #     neuron_model = LIFNeuron(params)
            
        #     # Extract the ideal firing rates for the MU
        #     r_ideal_mu = r_ideal_mat[mu_idx] 

        #     # Run the optimiziation
        #     opt_x, opt_neuron = self.optimizeNeuronProperties(excitation, r_ideal_mu, neuron_model, mu_idx)

        #     # Store outputs
        #     tau_d_output[mu_idx] = opt_x[0]
        #     size_output[mu_idx] = opt_x[1]

        # Optimize over each mu using a multithreading approach 
        import multiprocessing
        Nproc = self.params['Nproc']
        pool = multiprocessing.Pool(Nproc) 

        # Run the optimization for all MUs identified for this contraction
        results = np.array(pool.map(self.worker, range(len(r_ideal_mat))))

        # Unpack output
        tau_d_output, size_output, error_output = results[:,0], results[:,1], results[:,2]

        #  and return the optimized values
        return tau_d_output, size_output, error_output
    

    
    def worker(self, mu_idx): 
        ''' 
        Worker function to be called for multithreading implementation
        '''
        # Import the neuron model
        from Models.LIFNeuron import LIFNeuron

        # Generate a neuron to use in the optimization 
        neuron_model = LIFNeuron(self.params)
        
        # Extract the ideal firing rates for the MU
        r_ideal_mu = self.r_ideal_mat[mu_idx] 

        # Run the optimiziation (normal optimization over firing rate)
        opt_x, opt_fun, opt_neuron = self.optimizeNeuronProperties(self.excitation, r_ideal_mu, neuron_model, mu_idx)
        # Run the optimization (first optimize size to RT, then tau_d to firing rate)
        # opt_x, opt_fun, opt_neuron = self.optimizeNeuronPropertiesSplit(self.excitation, r_ideal_mu, neuron_model, mu_idx)
    
        # Dummy output for debugging
        # opt_x = (1,0)
        # opt_fun = 0.5

        # Store outputs
        tau_d_output = opt_x[0]
        size_output = opt_x[1]

        return tau_d_output, size_output, opt_fun