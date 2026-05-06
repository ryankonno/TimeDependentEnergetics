'''
This code is designed to investigate the time activation dynamics for a single MU/muscle fibre

Based on the code for the full MU model but stripped down to remove any scaling and multiple MU calculations

Ryan Konno
The University of British Columbia

'''
import numpy as np
from scipy.integrate import solve_ivp

# Import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

# Import parameters for the activation
from matplotlib import cm

import multiprocessing

###################################################################
# Implementation of activation model from Mayfield et al. 2022.
class ActivationModel():
    def __init__(self, params, t = None, plot_output = False):

        # Intialize the time constants
        self.tau_1 = params['Tau_1']
        self.tau_2 = params['Tau_2']

        self.params = params

        self.t = t

        self.plot_output = plot_output

    ###################################################################
    '''
    Function to run the single fibre/MU Ca dynamics 

    Input: 
        t_stim_idx: index of stimulation times
    '''
    def runExcAct(self, t_stim_idx, w_0 = 0.0048):

        # Now compute the Ca concentrations 
        ca_vec = self.computeCaConvolution(self.tau_1, self.tau_2, t_stim_idx, w_0 = w_0) 

        # Compute the bound Ca 
        catn_vec = self.hillEquation(ca_vec)

        # Compute the stimulation values 
        stim_vec = self.stim(t_stim_idx, w_0 = w_0)

        return stim_vec, ca_vec, catn_vec

    ###################################################################
    # Stimulus function from the pulses obtained from the MU data
    # This function needs to convert the single instantaneous pulses into the times at which the Ca
    # is released from the SR. This should take into account the attenuation of the pulse width 
    def stim(self, t_stim_idx, w_0 = 0.0048):
        t_n_1 = None
        stims = np.zeros(np.size(self.t)) # Get the the stimulus
        dt = self.t[1] - self.t[0]

        # Loop over MU firings
        # for MUn in self.MUAP_idx:
        # print(self.t[np.int64(MUAP_idx)])
        for stim_idx in t_stim_idx:
            t_n = self.t[stim_idx]
            # print(t_n)
            # Compute the stimulus values for the action potential 
            A = 0.2 # Values from Mayfield et al. 2022
            r = 0.35 # s, from Mayfield et al. 2022a
            if t_n_1 is None:
                # Use full base width for the first pulse (no prior ISI).
                width = w_0
            else:
                t_isi = t_n - t_n_1
                width = w_0 * (1 - A * np.exp(-t_isi/r)) # Compute the pulse width based on isi
                # width = w_0  # Compute the pulse width based on isi

            # Keep onset after t_n while ensuring at least one sample-wide pulse.
            width_eff = max(width, dt)
            stims = stims + (self.t > t_n) * (self.t <= t_n + width_eff)

            # Update previous firing time 
            t_n_1 = t_n

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
    
    def computeCaConvolution(self, tau_1_, tau_2_, MUAP_idx, w_0 = 0.0048):
        """
        Computes calcium levels using a convolution-based approach to match the analytic implementation.
        
        Parameters:
            tau_1_ (float): Time constant for calcium activation (rise).
            tau_2_ (float): Time constant for calcium decay (fall).
            MUAP_idx (array): Indices of stimulus events within self.t.
        
        Returns:
            numpy array: Calcium levels over time.
        """
        # Extract time vector
        t = self.t
        dt = t[1] - t[0]  # Time step size
        n = len(t)
        
        # Create stimulus train (binary array where stimuli occur)
        stim_train = np.zeros(n)
        stim_train[MUAP_idx] = 1

        # Define the impulse response for calcium dynamics
        def impulse_response(t, tau_1, tau_2):
            """
            Impulse response of the calcium system, combining activation and decay.
            """
            response = np.zeros_like(t)
            response[t >= 0] = (1 - np.exp(-t[t >= 0] / tau_1)) * np.exp(-t[t >= 0] / tau_2)
            return response

        # Generate the impulse response
        kernel_duration = 5 * max(tau_1_, tau_2_)  # Capture sufficient time for decay
        kernel_time = np.arange(0, kernel_duration, dt)
        kernel = impulse_response(kernel_time, tau_1_, tau_2_)

        # Convolve the stimulus train with the kernel
        ca_levels = np.convolve(stim_train, kernel, mode='full')[:n]

        # Match activation-decay behavior from the analytic function
        stim_times = t[MUAP_idx]
        a_curr = 0
        a = np.zeros_like(t)
        prev_idx = 0
        next_idx = 1

        for idx, t_ in enumerate(t):
            # Update interstimulus interval
            if next_idx < len(stim_times) and t_ > stim_times[next_idx]:
                prev_idx += 1
                next_idx += 1

            t_isi = stim_times[next_idx] - stim_times[prev_idx] if next_idx < len(stim_times) else np.inf

            # Check if in stimulation region
            A = 0.2  # Values from Mayfield et al. 2022
            r = 0.35  # s, from Mayfield et al. 2022a
            # w_0 = 0.0048
            width = w_0 * (1 - A * np.exp(-t_isi / r)) # Width decayse with successive pulses (mu firing)
            # width = w_0 # Fixed width (for stimulation studies)
            width_eff = max(width, dt)

            if (t_ <= stim_times[prev_idx] + width_eff) and (t_ > stim_times[prev_idx]):
                # If in stimulation region, update activation
                a_curr = 1 - (1 - a_curr) * np.exp(-dt / tau_1_)
            else:
                # If not in stimulation region, decay calcium
                a_curr = a_curr * np.exp(-dt / tau_2_)

            # Store the calcium level
            a[idx] = a_curr

        return a


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
            # width = w_0 
            dt = self.t[1] - self.t[0]
            width_eff = max(width, dt)

            if (t_n <= width_eff + t_stim) and (t_n > t_stim):
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

    ###################################################################
    # Hill equation used for computing the bound Ca 
    def hillEquation(self,a):
        K = self.params["K"]
        n = self.params["n"]
        return np.power(a,n)/(np.power(a,n) + K**n)
