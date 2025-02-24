# Code to analyze the Excitation-Activation dynamics
import numpy as np
from scipy.integrate import solve_ivp

# Import matplotlib
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

# Import parameters for the activation
from lib.MUScalingFunctions import muCSADistr, varMUDist
from matplotlib import cm

import multiprocessing

###################################################################
# Implementation of activation model from Mayfield et al. 2022.
class ActivationModel():
    def __init__(self, params, t = None, plot_output = False):
        # Intialize the parameters to be used
        self.tau_1 = 0
        self.tau_2 = 0

        self.params = params

        self.t = t

        self.plot_output = plot_output

        # Threshold to consider a MU active
        self.MUact_thresh = 0.9

        # Scale the activation constants with the MU type
        self.tau_1_scaled, self.tau_2_scaled = self.scaleTimeConstants()


    ###################################################################
    '''
    This function is to run the MU pool exctitation activation dynamics 
    '''
    def runMUPoolExcAct(self, t, DT):

        # Initialize variables
        self.t = t
        self.fsamp = int(1/(t[1] - t[0]))
        self.DT = DT
        self.N_MU = len(DT)

        # Using multithreading class 
        # Worker function is defined below
        # Initialize 
        Nproc = 10
        pool = multiprocessing.Pool(Nproc)

        # Run for each N_MU
        results = pool.map(self.worker, range(self.N_MU))

        # Close the multiprocessing pool
        pool.close()
        pool.join()

        self.N_act_mat, self.stims, self.Cavals  = zip(*results)

        # Restructure the array
        self.N_act_mat = np.vstack(self.N_act_mat)

        return np.transpose(self.N_act_mat)

    # Adapt above code for multithreading 
    def worker(self, MU_num):
        # print('________________________________')
        # print(f'MU Number: {MU_num}')
         # Get the time constant 
        tau_1_, tau_2_ = self.getTimeConstants(MU_num)

        # Set the MUAP indexes 
        MUAP_idx = np.nonzero(self.DT[MU_num])

        # Now compute the Ca concentrations 
        Cavals_MU = self.computeCaAnalytic(tau_1_, tau_2_, MUAP_idx) 

        # Now we can compute the level of activation 
        act_MU = self.getActivation(Cavals_MU)

        # Compute the stimulus values (takes into account isi)
        if self.plot_output:
            stims = self.stim(MUAP_idx)
        else: 
            stims = np.zeros_like(act_MU)

        return act_MU, stims, Cavals_MU

    '''
    Function to scale the time constants with contraction rate 
    '''
    def getTimeConstants(self, MUnum):

        tau_1_ = self.tau_1_scaled[MUnum]
        tau_2_ = self.tau_2_scaled[MUnum]

        return tau_1_, tau_2_

    '''
    Compute the scaling for the activation dynamics time constants
    '''
    def scaleTimeConstants(self):
        # Scaled for muscle composition of slow and fast fibres
        tau_1_slow = self.params['Activation_model']['Tau_1_slow']
        tau_2_slow = self.params['Activation_model']['Tau_2_slow']
        tau_1_fast = self.params['Activation_model']['Tau_1_fast']
        tau_2_fast = self.params['Activation_model']['Tau_2_fast']

        # Define a the array of MU numbers 
        mu_list = np.arange(self.params['N_MU_sim'])
        

        # Determine the scaling 
        # We want to find the MU number where there would be a change from slow to fast type fibres 
        #   this is assuming there is a hard switch between MU types (unlikely in reality)
        #   In a later step, we can account for a smoother transition
        muCSADistr_list = muCSADistr(mu_list, self.params)
        A_mu_dist_cumsum = np.cumsum(muCSADistr_list)
        slow_fibre_pcsa = self.params['muscle_pcsa'] * self.params['N_I'] 
        fibre_trans_point = int(np.argmin(np.abs(A_mu_dist_cumsum - slow_fibre_pcsa)))
        # print(f'Fibre-type transition point {fibre_trans_point}')

        # Calculate the scaled values
        tau_1_scaled = varMUDist(mu_list, fibre_trans_point, tau_1_slow, tau_1_fast, self.params)
        tau_2_scaled = varMUDist(mu_list, fibre_trans_point, tau_2_slow, tau_2_fast, self.params)

        if self.plot_output: 
            # Plot results to verify 
            # Plot the results
            fig, ax = plt.subplots(layout='constrained', figsize = (6,4))

            # First plot the normalized torque recruitment threshold 
            # ax.plot(mu_list, 100 * threshFunction(mu_list, params['N_MU_sim']), color='k', label= 'Rec Thresh')
            ax.plot(mu_list, muCSADistr_list, color='k', label= 'MU CSA Distribution')
            ax.plot((fibre_trans_point,fibre_trans_point), (0,max(muCSADistr_list)), 'k', label = 'Trans. Point')
            ax.set_xlabel('MU number')
            ax.set_ylabel('MU CSA (m^2)')

            # Plot the transition point between slow and fast fibres 
            ax2 = ax.twinx() 
            ax2.plot(mu_list, tau_1_scaled, color='r', label='Tau_act')
            ax2.set_ylabel('Time constant (s)', color='r')
            ax2.legend()        

            fig.savefig(self.params['figures_save_dir'] + '/TimeConstScaling_incr.pdf')
            plt.close()

            # Plot results to verify 
            # Plot the results
            fig, ax = plt.subplots(layout='constrained', figsize = (6,4))

            # First plot the normalized torque recruitment threshold 
            # ax.plot(mu_list, 100 * threshFunction(mu_list, params['N_MU_sim']), color='k', label= 'Rec Thresh')
            ax.plot(mu_list, muCSADistr_list, color='k', label= 'MU CSA Distribution')
            ax.plot((fibre_trans_point,fibre_trans_point), (0,max(muCSADistr_list)), 'k', label = 'Trans. Point')
            ax.set_xlabel('MU number')
            ax.set_ylabel('MU CSA (m^2)')

            # Plot the transition point between slow and fast fibres 
            ax2 = ax.twinx() 
            ax2.plot(mu_list, tau_2_scaled, color='r', label='Tau_deact')
            ax2.set_ylabel('Time constant (s)', color='r')
            ax2.legend()        

            fig.savefig(self.params['figures_save_dir'] + '/TimeConstScaling_decay.pdf')
            plt.close()

        return tau_1_scaled, tau_2_scaled

    ###################################################################
    # Get the excitation levels 
    # def getCaLevels(self,stims, tau_1_, tau_2_, MUAP_idx):
    #     # Get the Ca levels 

    #     # Solve numerically
    #     # return self.solveCaODE(stims, tau_1_, tau_2_)

    #     # Get the excitation levels using the solutions to the ODE 
    #     return self.computeCaAnalytic(stims, tau_1_, tau_2_, MUAP_idx)

    ###################################################################
    # Stimulus function from the pulses obtained from the MU data
    # This function needs to convert the single instantaneous pulses into the times at which the Ca
    # is released from the SR. This should take into account the attenuation of the pulse width 
    def stim(self,MUAP_idx):
        t_n_1 = 0
        stims = np.zeros(np.size(self.t)) # Get the the stimulus
        # print(self.MUAP_idx[0,:])

        # Loop over MU firings
        # for MUn in self.MUAP_idx:
        # print(self.t[np.int64(MUAP_idx)])
        for t_n in self.t[np.int64(MUAP_idx)][0]:
            # Get firing time  
            # t_n = self.t[MUn]
            # print(f'MUn = {MUn}')
            # print(f'Size of t_n = {np.size(t_n)}')
             
            # Compute the stimulus values for the action potential 
            # TODO: Determine appropriate pulse width 
            t_isi = t_n - t_n_1
            A = 0.2 # Values from Mayfield et al. 2022
            r = 0.35 # s, from Mayfield et al. 2022a
            w_0 = 0.0048
            width = w_0 * (1 - A * np.exp(-t_isi/r)) # Compute the pulse width based on isi
            # width = 0.0025* t_isi # Pulse width of 5ms
            
            # Compute the action potential
            # print(width)
            # AP = np.exp(-(self.t-t_n)**2/width) # Here width is the width of the action potential
            # AP = AP * (AP > 0.01) # Keep only AP values greater than 0.01
            # stims = AP + stims # Add the AP to the stimulus values

            stims = stims + (self.t > t_n) * (self.t < t_n + width)

            # Update previous firing time 
            t_n_1 = t_n

        # Plot the stimulus for verification of MUAP
        # plt.plot(self.t,stims)
        # # plt.savefig('Figures/Stimulus.pdf')
        # plt.show()
        # plt.close()

        return stims

    ###################################################################
    # Determine if the current time is a simulus
    # Return 1 if yes, 0 if no 
    def getStimVal(self,tval,stims ):
        # Interpolate the stimulus values at tval 
        # stimval = np.interp(tval,self.t,stims)
        stimval = stims[int(tval*self.fsamp)]

        # Return stimulus value of muscle 
        thresh = 0.95 # Value at which we consider the muscle to be stimulated

        return (stimval > thresh)

    # Analytically solve for the Ca levels 
    def computeCaAnalytic(self, tau_1_, tau_2_, MUAP_idx):
        # Note this is bad implementation
        # It may be possible to do this better with a convolution
        a_0 = 0
        t_0 = 0
        a_1 = 0 
        t_1 = 0
        instim = True
        a = np.zeros(np.shape(self.t))
        a_curr = 0
        idx = 0

        # print(f'Size of MUAP_idx: {np.shape(MUAP_idx)}')
        # print(f'Size of MUAP_idx: {MUAP_idx}')

        # Function to compute if we are in a stimulus region
        def computeStimRegion(t_isi, t_stim, t_n):
            A = 0.2 # Values from Mayfield et al. 2022
            r = 0.35 # s, from Mayfield et al. 2022a
            w_0 = 0.0048
            width = w_0 * (1 - A * np.exp(-t_isi/r))

            if (t_n < width + t_stim) and (t_n > t_stim):
                return True
            else: 
                return False

        # Define the stimulus times
        stim_times = self.t[MUAP_idx]
        # print(f'stim_times: {stim_times}')
        # print(f'Length of stim_times: {len(stim_times)}')

        if len(stim_times) > 0: 

            # Define the indices
            prev_idx = 0
            next_idx = 1

            # Loop over time points
            for t_ in self.t: 
                # Update the indices if we are past the previous time
                try:    
                    if t_ > stim_times[next_idx]:
                        prev_idx +=1 
                        next_idx +=1
                    
                    # Define the interstimulus interval
                    t_isi = self.t[int(next_idx)] - self.t[int(prev_idx)]
                except: 
                    if next_idx >= len(stim_times):
                        t_isi = 1e100 # Set so that we have the maximum width based on the isi relationship
                    else: 
                        print("ERROR!")


                # Check if we are in a simulus
                if computeStimRegion(t_isi, stim_times[prev_idx], t_): 
                    # If we are AND were not befor (instim = False),
                    # then we start to activate
                    # Update the parameters
                    if not instim:
                        instim = True # Update instim to true since we are in the stime
                        t_0 = self.t[idx-1] 
                        a_0 = a_curr
                    
                    # Solution to ODE from MCL (2023)
                    a_curr = 1 - (1-a_0) * np.exp(-(t_ - t_0)/tau_1_)
                else: 
                    # If we are no longer in a stimulus
                    # Then we update the constants for the decay of Ca
                    if instim:
                        # Update the parameters
                        instim = False
                        t_1 = self.t[idx-1]
                        a_1 = a_curr

                    # Solution to ODE from MCL (2023)
                    a_curr = max(a_1 * np.exp(-max((t_ - t_1)/tau_2_, 1e-10)), 0)
                
                # Add value to the list of Ca concentrations
                a[idx] = a_curr
                
                # Increment counter
                idx += 1

        return a
    
        # # Note this is bad implementation
        # # It may be possible to do this better with a convolution
        # a_0 = 0
        # t_0 = 0
        # a_1 = 0 
        # t_1 = 0
        # instim = True
        # a = np.zeros(np.shape(self.t))
        # a_curr = 0
        # idx = 0

        # # Loop over time points
        # for t_ in self.t: 
        #     # Check if we are in a simulus
        #     if self.getStimVal(t_,stims): 
        #         # If we are AND were not befor (instim = False),
        #         # then we start to activate
        #         # Update the parameters
        #         if not instim:
        #             instim = True # Update instim to true since we are in the stime
        #             t_0 = self.t[idx-1] 
        #             a_0 = a_curr
                
        #         # Solution to ODE from MCL (2023)
        #         a_curr = 1 - (1-a_0) * np.exp(-(t_ - t_0)/tau_1_)
        #     else: 
        #         # If we are no longer in a stimulus
        #         # Then we update the constants for the decay of Ca
        #         if instim:
        #             # Update the parameters
        #             instim = False
        #             t_1 = self.t[idx-1]
        #             a_1 = a_curr

        #         # Solution to ODE from MCL (2023)
        #         a_curr = max(a_1 * np.exp(-max((t_ - t_1)/tau_2_, 1e-10)), 0)
            
        #     # Add value to the list of Ca concentrations
        #     a[idx] = a_curr
            
        #     # Increment counter
        #     idx += 1

        # return a


    ###################################################################
    # Hill equation used for computing the activation dynamics
    def hillEquation(self,a):
        K = self.params['Activation_model']["K"]
        n = self.params['Activation_model']["n"]
        return np.power(a,n)/(np.power(a,n) + K**n)

    ###################################################################
    # Compute the activation based on the Ca levels
    # Using the implementation from Mayfield et al. 2023 (Hill equation)
    def getActivation(self, Cavals):
        return self.hillEquation(Cavals)

    ###################################################################
    # Plot the activation levels for one MU
    def plotActivation(self, mu_num):
        plt.figure(layout = 'constrained',figsize = (6,4))
        plt.plot(self.t,self.stims[int(mu_num)],label='Stim')
        plt.plot(self.t,self.Cavals[int(mu_num)],label='[Ca2+]')
        plt.plot(self.t,self.N_act_mat[int(mu_num)],label="[CaTn]")
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized values')
        plt.legend()
        # plt.show()
        plt.savefig(self.params['figures_save_dir'] + '/MU' + str(mu_num) + '_actdynamics.pdf')
        # plt.show()
        plt.close()

    ###################################################################
    # Function to get scaled activation levels 
    # Assume given a matrix 
    def scaleActivation(self, N_act_mat_):
        '''
        Scale the activation levels 
        '''
        # Define a list of MU numbers 
        mu_list = np.arange(self.params['N_MU_sim'])

        # First get the scaling values
        mu_csa_list = muCSADistr(mu_list, self.params)
        sum_csa_list = self.params['muscle_pcsa']

        # Define the scaling vector 
        # Will proportionally scale all of the activation levels so that the sum will be 1 if all MUs are active
        mu_scaling_list = mu_csa_list / sum_csa_list

        # Perform the scaling of the matrix 
        # print(f'Size of mu_scaling_list = {np.size(mu_scaling_list)}')
        # print(f'Size of N_act_mat_ = {np.size(N_act_mat_)}')
        # print(mu_scaling_list)
        # print(N_act_mat_)
        N_act_mat_scaled = N_act_mat_ * mu_scaling_list

        # Get a surface plot of  the activation levels across the 
        if self.plot_output:
            self.plotMUActivationScaling(N_act_mat_scaled)


        return N_act_mat_scaled
    
    ###################################################################
    def plotMUActivationScaling(self, N_act_mat_scaled): 
        '''
        Function for a surface plot of the activation levels 
        '''
        # Initialize the plot
        fig, ax = plt.subplots(layout = 'constrained',subplot_kw={"projection": "3d"})

        # Define the data
        x = np.arange(self.params['N_MU_sim'])
        x = x[0::5]
        y = self.t[0::10]
        X, Y = np.meshgrid(x, y)
        
        surf = ax.plot_surface(X, Y, N_act_mat_scaled[0::10,0::5], cmap=cm.hot, linewidth=0, antialiased=False)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_xlabel('MU number')
        ax.set_ylabel('Time(s)')
        ax.set_zlabel('Activation (normalized)')
        
        ax.azim = -120

        plt.savefig('Figures/' + self.params['run_id'] + '/ActivationSurface.pdf')
        plt.close()
        
        return None
