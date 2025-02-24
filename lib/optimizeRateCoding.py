# Include 
import numpy as np
from lib.smoothDT import smoothDT
from scipy.optimize import minimize_scalar, minimize
import multiprocessing

import matplotlib.pyplot as plt
import matplotlib 
font = {'size'   : 12}
matplotlib.rc('font', **font)
import time

# Import parameters
# from parameters import params 

class MUOptimizer():
    def __init__(self, lif_model,  time, DT, exc, fsamp, exc_thresh_sim, n_mu_sim_pop, params): 

        # Initialize the variables
        self.time = time 
        self.DT = DT 
        self.exc = exc 
        self.fsamp = fsamp 
        self.exc_thresh_sim = exc_thresh_sim
        self.n_mu_sim_pop = n_mu_sim_pop
        self.params = params
        self.lif_mu_model_exp = lif_model

        return None 

    def worker(self, n_mu): 
        ''' 
        Function added to multithread the optimization of each MU below 
        '''
        import numpy as np
        from scipy.optimize import minimize_scalar, minimize

        # Get MU number in full population 
        n_mu_sim = int(self.n_mu_sim_pop[n_mu])
        
        # Get the simulated MU number 
        # n_mu_sim = int(n_mu_sim_pop[n_mu])

        # Get smoothed rate experimental rate coding
        r_exp_smoothed = smoothDT(self.time, self.DT[n_mu], self.fsamp, n_mu, self.params['run_id'])

        # Now define the objective function 
        # TODO: Adapt this for the possibility of more than one parameter
        def J(mu_size): 
            _, _, r_sim_cts  = self.lif_mu_model_exp.r_sim(n_mu_sim, mu_size)
            return np.sqrt(np.mean((r_sim_cts - r_exp_smoothed)**2))/ np.sqrt(np.mean((r_exp_smoothed)**2))
        
        size_min = self.params['MU_model']['mu_size_min']
        size_max = self.params['MU_model']['mu_size_max']

        opt_res = minimize_scalar(J, bracket = (size_min, size_max), bounds = (size_min, size_max), method = 'bounded', options={'xatol':1e-8})

        MU_sizes = opt_res.x

        print('________________________________________')
        print(f'MU Number: {n_mu}')
        print(f'MU number in sim pool: {n_mu_sim}')
        print(f'Optimized MU size: {opt_res.x}')
        print(f'Optimized objective: {opt_res.fun}')

        # Plot the optimized rates
        fig,ax = plt.subplots(layout = 'constrained',figsize=(6, 4))
        ax.plot(self.time, r_exp_smoothed, label='Experimental')
        _, _, r_sim_cts = self.lif_mu_model_exp.r_sim(n_mu_sim, opt_res.x)
        ax.plot(self.time, r_sim_cts, label='Simulated')
        ax.legend(loc='upper left')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing rate (Hz)')
        ax2 = ax.twinx()
        ax2.plot(self.time, self.exc, 'k', ls = 'dashed', label = 'Excitation')
        ax2.set_ylabel('Excitation (Normalized)')
        ax2.legend(loc='upper right')
        if self.params['save_data']: 
            plt.savefig('./Figures/' + self.params['run_id'] + '/MU' +str(n_mu) + '_fit.pdf')
        plt.close()

        return MU_sizes

    def optimizeRateCoding(self):
        '''
        This function is designed to optimize the MU rate coding parameters

        For now we assume a linear function but there may need to be adjustments to make the function nonlinear.

        Input: 
            time: time vector
            DT: Experimental discharge times for all MUs
            exc: Common excitation to the MU pool
            fsamp: sample frequency
            exc_thresh_sim: MU excitation thresholds for the simulated population
            n_mu_sim_pop: Corresponded experimental MUs in simulated population

        Output: 
            rc_params: Rate coding parameters

        Ryan Konno
        '''

        if self.params['MU_model']['Type'] == 'Fuglevand':
            # Import the MU model 
            from OLD.MUModel import r_sim

            # TODO: generalize this 
            if self.params['MU_model']['rate_coding_fun'] == 'log':
                gain = np.zeros((np.size(self.DT),3))
            else:
                gain = np.zeros(np.shape(self.DT))

            # Original 
            # Loop over the MUs
            for n_mu in range(np.size(self.DT)):
                print('________________________________________')
                print(f'MU Number: {n_mu}')

                # Get MU number in full population
                n_mu_sim = int(self.n_mu_sim_pop[n_mu]) 

                # Get smoothed rate experimental rate coding
                r_exp_smoothed = smoothDT(time, self.DT[n_mu], self.fsamp)

                # Now define the objective function 
                # TODO: Adapt this for the possibility of more than one parameter
                def J(g): 
                    _, r_sim_cts  = r_sim(time, self.exc, g, n_mu_sim, self.exc_thresh_sim[n_mu_sim], self.exc_thresh_sim[-1], self.params)
                    return np.sum((r_sim_cts - r_exp_smoothed)**2)/ np.sum((r_exp_smoothed)**2)

                # Perform the optimization
                # Save the gain value 
                if self.params['MU_model']['rate_coding_fun'] == 'log': 
                    g0 = (0.1, 1.5, 2.4) # Initial starting values for optimization
                    opt_res = minimize(J, g0, options={'maxiter':100, 'disp': False})
                    gain[n_mu,:] = opt_res.x
                else: 
                    opt_res = minimize_scalar(J, options={'maxiter':100, 'disp': False})
                    gain[n_mu] = opt_res.x
                
                print(f'Optimized gain: {opt_res.x}')
                print(f'Optimized objective: {opt_res.fun}')

                # Plot the optimized rates
                # fig,ax = plt.subplots(layout = 'constrained',figsize=(10, 6))
                # ax.plot(time, r_exp_smoothed, label='Experimental')
                # _, r_sim_cts = r_sim(time, exc, opt_res.x, n_mu_sim, exc_thresh_sim[n_mu_sim], exc_thresh_sim[-1], params)
                # ax.plot(time, r_sim_cts, label='Simulated')
                # ax.legend(loc='upper left')
                # ax.set_xlabel('Time (s)')
                # ax.set_ylabel('Firing rate (Hz)')
                # ax2 = ax.twinx()
                # ax2.plot(time,exc, ls = 'dashed', label = 'Excitation') 
                # ax2.set_ylabel('Excitation (Normalized)')
                # ax2.legend(loc='upper right')
                # if params['save_data']: 
                #     plt.savefig('./Figures/' + params['run_id'] + '/MU' +str(n_mu) + '_fit.pdf')
                # plt.show()
            output_param = gain
            
        elif self.params['MU_model']['Type'] == 'LIF':
            # # Initialize the LIF model
            # from Models.LIFModel import LIFModel
            # self.lif_mu_model_exp = LIFModel(self.time, self.DT, self.exc_thresh_sim, self.exc, self.params)

            # Define array to store values 
            MU_sizes = np.zeros(np.shape(self.DT))
            
            # Optimize all MU sizes
            # Multithreaded version 
            Nproc = 10
            pool = multiprocessing.Pool(Nproc)

            # Run the model for each MU
            start_time = time.time()        
            results = pool.map(self.worker, range(np.size(self.DT)))
            output_param = np.array(results)
            # output_param = zip(*results) % If returning more than one value from the worker function
            end_time = time.time()

            elapsed_time = end_time - start_time
            print(f"Time to optimize the MU pool: {elapsed_time:.2f} seconds")

            pool.close()
            pool.join()

            # # Loop over the MUs and optimize the MU size 
            # # TODO: Finish implementing this section (see Caillet implementation for help with this)
            # for n_mu in range(np.size(self.DT)):
            #     print('________________________________________')
            #     print(f'MU Number: {n_mu}')

            #     # Get MU number in full population 
            #     n_mu_sim = int(self.n_mu_sim_pop[n_mu])
            #     print(f'MU number in sim pool: {n_mu_sim}')

            #     # Get the simulated MU number 
            #     # n_mu_sim = int(n_mu_sim_pop[n_mu])

            #     # Get smoothed rate experimental rate coding
            #     r_exp_smoothed = smoothDT(self.time, self.DT[n_mu],self.fsamp)

            #     # Now define the objective function 
            #     # TODO: Adapt this for the possibility of more than one parameter
            #     def J(mu_size): 
            #         _, _, r_sim_cts  = self.lif_mu_model_exp.r_sim(n_mu_sim, mu_size)
            #         return np.sqrt(np.mean((r_sim_cts - r_exp_smoothed)**2))/ np.sqrt(np.mean((r_exp_smoothed)**2))
                
            #     size_min = self.params['MU_model']['mu_size_min']
            #     size_max = self.params['MU_model']['mu_size_max']


            #     # TEST
            #     # Plot the effect of MU size...
            #     # fig,ax = plt.subplots(layout='constrained',figsize=(10, 6))
            #     # for mu_size in np.linspace(size_min, size_max, 10): 
            #     #     # Get the MU size 
            #     #     _, r_sim_cts = lif_mu_model_exp.r_sim(n_mu_sim,mu_size) 
            #     #     ax.plot(time, r_sim_cts, label = 'MU size = ' + str(mu_size))
            #     # ax.plot(time,r_exp_smoothed, color='k', label='Experimental')
            #     # ax2 = ax.twinx()
            #     # ax2.plot(time, exc, 'k', ls='dashed',label='Excitation')
            #     # ax.legend(loc='upper left')
            #     # ax2.legend(loc='upper right')
            #     # ax.set_xlabel('Time (s)') 
            #     # ax.set_ylabel('Firing Rate (Hz)')
            #     # ax2.set_ylabel('Excitation (Normalized)')
            #     # if params['save_data']: 
            #     #     plt.savefig('./Figures/' + params['run_id'] + '/MU' + str(n_mu) + 'VaryMUSize_fit.pdf')
            #     # plt.show()



            #     # opt_res = minimize_scalar(J, bracket=(size_min, size_max), bounds=(size_min, size_max), method='bounded', options={'xatol':1e-9})
            #     opt_res = minimize_scalar(J, bracket=(size_min, size_max), bounds=(size_min, size_max), method = 'bounded', options={'xatol':1e-8})
            #     # options={'xatol':1e-9} # Option from Caillet 2022

            #     MU_sizes[n_mu] = opt_res.x

            #     print(f'Optimized MU size: {opt_res.x}')
            #     print(f'Optimized objective: {opt_res.fun}')

            #     # Plot the optimized rates
            #     # fig,ax = plt.subplots(layout = 'constrained',figsize=(10, 6))
            #     # ax.plot(time, r_exp_smoothed, label='Experimental')
            #     # _, r_sim_cts = lif_mu_model_exp.r_sim(n_mu_sim, opt_res.x)
            #     # ax.plot(time, r_sim_cts, label='Simulated')
            #     # ax.legend(loc='upper left')
            #     # ax.set_xlabel('Time (s)')
            #     # ax.set_ylabel('Firing rate (Hz)')
            #     # ax2 = ax.twinx()
            #     # ax2.plot(time,exc, 'k', ls = 'dashed', label = 'Excitation') 
            #     # ax2.set_ylabel('Excitation (Normalized)')
            #     # ax2.legend(loc='upper right')
            #     # if params['save_data']: 
            #     #     plt.savefig('./Figures/' + params['run_id'] + '/MU' +str(n_mu) + '_fit.pdf')
            #     # plt.show()

            # output_param = MU_sizes
            

        # Plot the gain values 
        # fig,ax = plt.subplots(layout='constrained')
        # ax.plot(range(np.size(gain[:,1])), gain[:,0])
        # ax.plot(range(np.size(gain[:,1])), gain[:,1])
        # ax.set_ylabel('Gain (Normalized??)')
        # ax.set_xlabel('MU Number')
        # plt.show()

        return output_param
