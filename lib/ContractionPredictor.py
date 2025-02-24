'''
This is a class designed to simulate a predictive contraction
'''
import time as timer
import numpy as np
import matplotlib.pyplot as plt

class ContractionPredictor(): 
    def __init__(self, params): 
        self.params = params
    
    def defineExcitation(self, t_vec, excitation): 
        '''
        Define a vector or excitations 
        '''
        self.t_vec = t_vec
        self.excitation = excitation 

    def simulateMUPool(self, full_mu_array, tau_d_array_full, size_array_full): 
        '''
        Function to simulate the MU pool dynamics
        '''

        # Create the MU Pool model 
        from Models.LIFNeuronPool import LIFNeuronPool
        lif_pool_model = LIFNeuronPool(full_mu_array, tau_d_array_full, size_array_full, self.params['MU_model'])

        # Run the LIF pool model with the excitation
        # print('______________________________________')
        # print('Simulating MU pool...')
        start_time = timer.time()
        self.dt_mat = lif_pool_model.simulateMUPoolContraction(self.excitation, self.params['fsamp'])
        # print(f'Time elapsed = {timer.time() - start_time}')
        # print('______________________________________')

        # Return the matrix of discharge times
        return self.dt_mat

        

    def simulateExcAct(self, dt_mat = None):
        '''
        Function to simulate the Excitation-activation dynamics

        Input
            - optionally define the dt matrix at input (or use one already stored in class)
        '''
        if dt_mat == None and hasattr(self, 'dt_mat'): 
            dt_mat = self.dt_mat
        else: 
            raise ValueError('No dt_mat defined!!')


        # Import the activation class
        from Models.ActivationModel import ActivationModel
        # Initialize and compute necessary variables
        t_vec_sim = np.arange(0, self.params['MU_model']['t_end'], self.params['MU_model']['dt']) 

        # Generate a boolean discharge matrix (necessary for ActivationModel)
        dt_mat_bool = np.empty(len(self.dt_mat), dtype='object')
        for idx, dt_mu in enumerate(self.dt_mat):
            
            dt_mu_bool = np.zeros_like(t_vec_sim)
            
            # Set all discharge times equal to 1
            dt_mu_bool[dt_mu.astype(int)] = 1 

            # Add to matrix 
            dt_mat_bool[idx] = dt_mu_bool

        act_model = ActivationModel(self.params, t_vec_sim, not self.params['suppress_output']) # Initialize the activation model 

        # These activation levels are not normalized to location in the MU pool
        # print('Running the activation model...')
        # start_time = timer.time()
        self.act_mat = act_model.runMUPoolExcAct(t_vec_sim, dt_mat_bool)

        # end_time = timer.time()
        # elapsed_time = end_time - start_time
        # print(f"    Time to run the activation dynamics: {elapsed_time:.2f} seconds")

        # Plot the activation for one MU 
        # act_model.plotActivation(5) # Choose arbitrarily the 5th MU
        # act_model.plotActivation(300) # Choose arbitrarily the 5th MU


        # Perform scaling using ActivationModel function 
        self.act_mat_scaled = act_model.scaleActivation(self.act_mat)

        # Compute CSA values 
        from lib.MUScalingFunctions import muCSADistr
        self.A_mu_vec = muCSADistr(np.arange(self.params['N_MU_sim']), self.params)

        # Compute sigma_0_i values for the MUs 
        from lib.MUScalingFunctions import varMUDist
        self.sigma_0_i_vec = varMUDist(np.arange(self.params['N_MU_sim']), 1, self.params['sigma_0_slow'], self.params['sigma_0_fast'], self.params)
        # sigma_0_i_vec = alpha_s_i_vec * self.params['sigma_0_slow'] + (1 - alpha_s_i_vec) * self.params['sigma_0_fast'] # Can delete (above line calculates this)

        # Compute total activation as the sum of the number of MUs active at a give time 
        # (based on scaling above) 
        # NOTE: this global activation value has been scaled to account for different sigma_0 in MUs
        # TODO: VERIFY SCALED ACTIVATIONS ARE WORKING CORRECTLY
        self.act = np.zeros(np.shape(t_vec_sim))
        for n in range(np.size(t_vec_sim)): 
            # self.act[n] = sum(self.act_mat_scaled[n,:]) # Previous version not accounting for different MU sigma_0s
            self.act[n] = sum(self.sigma_0_i_vec * self.A_mu_vec * self.act_mat[n,:]) / (self.params['muscle_pcsa'] * self.params['sigma_0'])
        
        # Return the total activation, mu wise, and scaled mu wise activation
        return self.act, self.act_mat, self.act_mat_scaled



    def simulateMech(self, act = None, act_mat_scaled = None): 
        '''
        Function to simulate a Hill-type model for the MU pool as defined by previous call of simulateExcAct
        '''
        # Define variables
        if act == None and hasattr(self, 'act'): 
            act = self.act
        else: 
            raise ValueError('No dt_mat defined!!')
        if act_mat_scaled == None and hasattr(self, 'act_mat_scaled'): 
            act_mat_scaled = self.act_mat_scaled
        else: 
            raise ValueError('No dt_mat defined!!')
        
        # Initialize and compute necessary variables
        t_vec_sim = np.arange(0, self.params['MU_model']['t_end'], self.params['MU_model']['dt']) 

        # Define the model 
        from Models.MechanicsModel import MechModel
        mech_model = MechModel(self.params)

        # First define and set the activation vector over the time range 
        mech_model.setActivation(t_vec_sim, act) 
        mech_model.setActivationScalings(act_mat_scaled)

        # Run the mechanical model 
        # print('Solving the model mechanics...')
        # t_start = timer.time()
        t_vec_mech, e_m_, force_m = mech_model.solverMUPool(t_vec_sim[-1])
        # t_end = timer.time() 
        # print(f'    Time to solve model: {t_end - t_start:.2f} s')

        # Compute the strain rates 
        dedt_m_= np.diff(e_m_) / np.diff(t_vec_mech)

        # Interpolate all of the above values to self.t_vec 
        self.e_m = np.interp(self.t_vec, t_vec_mech, e_m_)
        self.dedt_m = np.interp(self.t_vec, t_vec_mech[0:-1], dedt_m_)
        self.force_m = np.interp(self.t_vec, t_vec_mech, force_m)


        # Plot the output 
        if not self.params['suppress_output']:
            mech_model.plotMechanics()
            mech_model.plotFVParamScaling(self.params['run_id'])

        return self.t_vec, self.e_m, self.force_m
        
    def simulateEnergetics(self, act_mat = None, e_m = None, dedt_m = None, force_m = None): 
        '''
        Function to simulate a Hill-type model for the MU pool as defined by previous call of simulateExcAct
        '''
        if hasattr(self, 'force_m'):
            act_mat = self.act_mat
            e_m = self.e_m
            dedt_m = self.dedt_m
            force_m = self.force_m
        elif force_m.any() == None:
            raise ValueError('No variables defined!!') 
            

        # Create the energetics model 
        from Models.EnergeticsModel import EnergeticsModel
        energetics_model = EnergeticsModel()

        
        # Get the energetics parameters 
        r1_scaled, r2_scaled = energetics_model.getEnergeticsConsts(self.params, not self.params['suppress_output'])
        self.params['r1'] = r1_scaled
        self.params['r2'] = r2_scaled

        # Define the maximum isometric force 
        # self.params['F_0'] = self.params['sigma_0'] * self.params['muscle_pcsa'] # Old assuming we scale activation
        # from lib.MUScalingFunctions import muCSADistr
        # F_0_mu = self.params['sigma_0'] * muCSADistr(np.arange(self.params['N_MU_sim']), self.params) # Assume we use full act levels (not scaled)
        F_0_mu = self.sigma_0_i_vec * self.A_mu_vec # Assume we use full act levels (not scaled)

        # For now, lets just compute the energetics for each MU individually 
        # TODO: Should be able to do this simultaneously
        total_energy_mat = np.zeros_like(act_mat)
        total_heat_mat = np.zeros_like(act_mat)
        act_heat_mat = np.zeros_like(act_mat)
        sl_heat_mat = np.zeros_like(act_mat)
        work_heat_mat = np.zeros_like(act_mat)
        for mu in np.arange(self.params['N_MU_sim']):
            self.params['F_0'] = F_0_mu[mu]
            # Use scaled MU properties
            heat_rate_dict_mu, heat_dict_mu  = energetics_model.dHdt(act_mat[:,mu], self.t_vec, e_m, dedt_m, force_m, r1_scaled[mu], r2_scaled[mu], self.params)

            # Use equal fibre-type distribution
            # alpha_s = self.params['alpha_s']
            # r1 = alpha_s * self.params['energetics_model']['r1_min'] + (1 - alpha_s) * self.params['energetics_model']['r1_max']
            # r2 = alpha_s * self.params['energetics_model']['r2_min'] + (1 - alpha_s) * self.params['energetics_model']['r2_max']
            # heat_rate_dict_mu, heat_dict_mu  = energetics_model.dHdt(act_mat[:,mu], self.t_vec, e_m, dedt_m, force_m, r1, r2, self.params)

            # Sort the data
            total_energy_mat[:,mu] = heat_rate_dict_mu['dEdt']
            total_heat_mat[:,mu] = heat_rate_dict_mu['dQdt']
            act_heat_mat[:,mu] = heat_rate_dict_mu['dQ_mdt']
            sl_heat_mat[:,mu] = heat_rate_dict_mu['dQ_sldt']
            work_heat_mat[:,mu] = heat_rate_dict_mu['dWdt'] 

        # Plot the heat rates 
        if not self.params['suppress_output']:
            energetics_model.plotHeatRates(self.t_vec, total_energy_mat, self.params)
            energetics_model.plotOverallHeatRate(self.t_vec, total_energy_mat, self.params)
            energetics_model.plotIndividualHeatRates(self.t_vec, act_heat_mat, sl_heat_mat, work_heat_mat, self.params)

            # Compare to the overall heat rates predicted 
            ave_heat_rate_dict, ave_heat_dict  = energetics_model.dHdt(self.act, self.t_vec, e_m, dedt_m, force_m, r1_scaled[0], r2_scaled[0], self.params)
            fig, ax = plt.subplots(layout='constrained',figsize = (6,4)) 
            ax.plot(self.t_vec, ave_heat_rate_dict['dQdt'], color='k', label='Overall Act (slow rates)')
            ax.plot(self.t_vec, np.sum(total_heat_mat, axis = 1), color='k', ls = 'dashed', label='MU Model')
            ax.legend()
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Energy rate (W)')
            plt.savefig('Figures/' + self.params['run_id'] + '/EnergeticsAverageComp.pdf')
            plt.close()

        # Combine total heat rates into one dict for output 
        heat_rate_dict = {
            'total_energy_mat': total_energy_mat,
            'total_heat_mat': total_heat_mat, 
            'act_heat_mat': act_heat_mat, 
            'sl_heat_mat': sl_heat_mat, 
            'work_heat_mat': work_heat_mat, 
        }

        return heat_rate_dict