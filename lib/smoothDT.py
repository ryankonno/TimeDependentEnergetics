def smoothDT(time, DT, fsamp, n_mu = 10000, run_id = ''):
    ''' 
    Function to compute the smoothed continuous discharge rates
    
    Assume that this only takes as input on single array of DT
        ie. from data from one MU

    Adapted from Caillet et al. 2022

    Input: 
        time: time array 
        DT: discharge times
        fsamp: sampling rate


    Output: 
        Smoothed firing times at each time point in *time*
    '''

    # 
    import numpy as np 
    from scipy.interpolate import CubicSpline,UnivariateSpline
    import matplotlib.pyplot as plt
    import os.path

    # First compute the discharge rates at each discharge time 
    r_at_DT = fsamp / (DT[1:] - DT[0:-1]) 

    # Get the discharge rates at all the time points

    # First define the values at time points 
    r_noncts = np.zeros(np.shape(time))
    r_noncts[np.array(DT[1:], dtype=int)] = r_at_DT


    # Get an array of ones at a discharge time and 0 otherwise 
    t_DT_binary = np.zeros_like(time) 
    DT_idxs = np.array(DT, dtype = 'int')
    t_DT_binary[DT_idxs] = 1

    #_______________________________
    # # Use cubic spline 
    # # Update the end values so the rates start and end at zero
    # # Pad the rates
    # r_splinevals = np.zeros((np.size(r_at_DT,0)+2,))
    # r_splinevals[1:-1] = r_at_DT 

    # # Combine into one array
    # # init_val = 
    # init_val = np.array([DT[0] - int(0.01*fsamp)])
    # final_val =  np.array([DT[-1] + int(0.01*fsamp)])
    # DT_spline = np.concatenate((init_val, DT[1:],final_val))# Pad the DT
    
    # # spline = CubicSpline(time[np.array((DT[1:]), dtype=int)], r_at_DT)
    # # spline = CubicSpline(time[np.array((DT_spline), dtype=int)], r_splinevals,bc_type=((1, 0.0), (1, 0.0)))
    # spline = UnivariateSpline(time[np.array((DT_spline), dtype=int)], r_splinevals,bc_type=((1, 0.0), (1, 0.0)))

    # # Define the continuous values 
    # r_cts = spline(time)

    # # Set rates outside of the contraction to zero
    # r_cts[time < time[init_val]] = 0
    # r_cts[time > time[final_val]] = 0
    
    #_______________________________
    # Use the Hanning filter from Caillet et al. 2022 
    # TODO: this implementation avoids the overfitting, but 
    #       there seems to be an issue with the scaling(se)
    from lib.HanningFilter import hanningFilter
    # r_cts = hanningFilter(r_noncts,fsamp) # What I was doing before
    r_cts = hanningFilter(t_DT_binary,fsamp) # Using a vector of ones and zeros
    # r_cts = r_cts/max(r_cts) * max(r_noncts) # Scaling !!! Need to figure out why !!!

    run_id = run_id + '/'
    file_name = 'Figures/' + run_id + 'MU' + str(n_mu) + '_firing_filt.pdf'
    
    # Plot to check interpolation 
    # if not os.path.isfile(file_name):
    #     print(f'Plotting firing rates mu: {n_mu}')
    #     fig,ax = plt.subplots(layout='constrained') 
    #     ax.plot(time, r_noncts, label = 'Unfiltered') 
    #     ax.plot(time, r_cts, label = 'Filtered') 
    #     ax.set_xlabel('Time (s)')
    #     ax.set_ylabel('Firing frequency (Hz)')
    #     ax.legend()
    #     plt.savefig(file_name)
    #     plt.close()

    return r_cts