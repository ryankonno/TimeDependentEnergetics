def getPeakFiringRate(t, DTs, fsamp, N_MU_sim, mu_sim_loc, force, run_id): 
    ''' 
    Code to calculate the peak firing rate of a selection of MUs 

    Ryan Konno
    '''
    # Import statements 
    import numpy as np 
    import matplotlib.pyplot as plt
    from lib.smoothDT import smoothDT
    from scipy.optimize import curve_fit

    # r_smoothed = np.zeros(np.shape(DTs))
    r_peak = np.zeros(np.size(DTs))

    # Remove the first second from the analysis 
    reg_to_remove = 1 * fsamp # Region to remove 
    force_cropped = force[reg_to_remove:]
    t_cropped = t[reg_to_remove:]

    # Get indices where force near maximal (ie plateau)
    #  THIS MAY NOT WORK FOR ALL MUS SO WE CAN CHOOSE THE PLATEAU BASED ON THE DISCHARGE RATES
    #       OR WE MAY NEED TO ONLY DO THIS WITH THE CONTRACTIONS AT HIGH INTENSITIES 
    # This process assumes a smooth torque trace (no large dips)
    # max_force = max(force_cropped[1*fsamp:]) # Get the max force 
    # idxs_near_f_max = np.array(force_cropped > 0.9*max_force) 
    # non_zero_indices = np.nonzero(idxs_near_f_max)[0] # Get the non zero indices (force is close to max)
    # # Define the indices for the region of interest
    # idx_f_max_start, idx_f_max_end = non_zero_indices[0], non_zero_indices[-1] 

    for mu in range(np.size(DTs)):
        # Calculate the smoothed firing rates of each MU
        r_smoothed_uncropped = smoothDT(t, DTs[mu], fsamp)

        # Here we also crop the discharge rates the same as the torque 
        r_smoothed = r_smoothed_uncropped[reg_to_remove:]

        #__________________________
        # Use force plateau
        # Save the peak firing rates of each MU
        # Take the average only if it is within 25% of the max
        # r_mean_val = np.mean(r_smoothed[idx_f_max_start:idx_f_max_end]) # Use with force plateau
        # r_max_val = max(r_smoothed[idx_f_max_start:idx_f_max_end])      # Use with force plateau

        #__________________________ 
        # Use discharge rate plateau
        # Now get the region of the discharge rates where a plateau has been reached 
        #   THIS IS AN OPTION INSTEAD OF CHOOSING BASED ON WHERE THE FORCE IS LARGEST
        #           WE MAY NEED TO THINK MORE ABOUT WHICH OPTION TO CHOOSE HERE
        r_max_val = max(r_smoothed) # Get the absolute max force ingoring
        idxs_near_r_max = np.array(r_smoothed > 0.9*r_max_val) 
        non_zero_indices = np.nonzero(idxs_near_r_max)[0] 
        idx_f_max_start, idx_f_max_end = non_zero_indices[0], non_zero_indices[-1] # Indices for region of interest
        r_mean_val = np.mean( r_smoothed[non_zero_indices]) # Compute mean value 

        # OPT1: Ensure that we are within 25% of the maximum value 
        # if r_mean_val > 0.75 * r_max_val: 
        #     r_peak[mu] = r_mean_val
        # else: 
        #     r_peak[mu] = np.nan
        # OP2: Take all peak firing rates 
        r_peak[mu] = r_mean_val

        print(f'r_peak of MU {mu} is {r_peak[mu]} Hz')
         
        # Plot the peak firing rate for verification (comment out after)
        fig,ax = plt.subplots() 
        ax.plot(t_cropped, r_smoothed, label = 'MU firing rate')
        ax.plot((t_cropped[idx_f_max_start],t[idx_f_max_start]), (0, max(r_smoothed)), color = 'k')
        ax.plot((t_cropped[idx_f_max_end],t[idx_f_max_end]), (0, max(r_smoothed)), color = 'k')

        ax.plot((t[0], t[-1]), (r_peak[mu], r_peak[mu]), label='Calculated r_peak')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Firing rate (Hz)')
        plt.savefig('Figures/' + run_id + '/MU' +str(mu) + '_r_peak.pdf')
        plt.close('all')
        


    # FITTING METHOD 1: Linear extraplolation to full MU pool
    # coef = np.polyfit(mu_sim_loc, r_peak,1)
    # poly1d_fn = np.poly1d(coef)   
    # r_peak_sim_pop = poly1d_fn(np.arange(N_MU_sim))
    # # Plot the peak firing rates for each MU
    # fig,ax = plt.subplots(layout='constrained') 
    # ax.plot(np.arange(N_MU_sim),r_peak_sim_pop) 
    # ax.plot(mu_sim_loc, r_peak, marker = 'o', color='k', ls = 'None')
    # ax.set_xlabel('Motor unit number')
    # ax.set_ylabel('Peak firing rate (Hz)')
    # plt.savefig('Figures/' + run_id + '/r_peak_plot.pdf')
    # plt.show()

    # FITTING METHOD 2: Power trendline 
    def power_dist(mu_nums, a, b): 
        return a * 10 * ((mu_nums/N_MU_sim) ** b)
    r_peak_clean = r_peak[~np.isnan(r_peak)]
    mu_sim_loc_clean = mu_sim_loc[~np.isnan(r_peak)]
    optimized_coeffs, pcov = curve_fit(power_dist, mu_sim_loc_clean, r_peak_clean, p0 = (1,1), method = 'lm')
    print(f'MU peak firing rate fit...')
    print(f'    Condition of fit: {np.linalg.cond(pcov)}')
    print(f'    Covariance diagonal: {np.diag(pcov)}')
    print(f'    Covariance matrix: {pcov}')
    print(f'    Parameters: {optimized_coeffs}')
    # Generate the simulated population 
    r_peak_sim_pop = power_dist(np.arange(N_MU_sim), *optimized_coeffs)
    # Plot the firing rate distribution 
    # fig,ax = plt.subplots(layout='constrained') 
    # ax.plot(np.arange(N_MU_sim), r_peak_sim_pop) 
    # ax.plot(mu_sim_loc, r_peak, marker = 'o', color='k', ls = 'None')
    # ax.set_xlabel('Motor unit number')
    # ax.set_ylabel('Peak firing rate (Hz)')
    # plt.savefig('Figures/' + run_id + '/r_peak_plot.pdf')
    # plt.show()



    return r_peak_sim_pop, r_peak