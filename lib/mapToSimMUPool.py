def mapToSimMUPool(RTs, N_MU_sim):
    '''
    Determine the locations of the experimentally measured MUs in
    the simulated MU pool. 
    
    Returns the indexes in the real MU pool
    
    Adapted from Caillet et al. 2022
    '''
    import numpy as np
    from lib.threshFunction import threshFunction
    
    # Initialize
    sim_mu_loc = np.zeros(np.size(RTs), dtype='int') # Specify int to use as an index

    # Loop over existing MUs
    for i in range (np.size(RTs)):
        # Set the index based on the argmin for abs of the difference between exp thresh and sim thresh
        sim_mu_loc[i] = int((  np.abs( threshFunction(np.arange(N_MU_sim), N_MU_sim) - RTs[i] )  ).argmin() )
        
        #looking for the closest match between typical and experimental %MVC recruitment threshold

    return sim_mu_loc
