def threshFunction(MU, N_MU_sim):
    '''
    Function for MU thresholds
    Input: 
        - MU: MU or list of MUs
        - N_MU_sim: total number of MUs in the simulated population
    TODO: add an option that allows for choosing the MU pool threshold function type
    '''
    # This function is from Caillet et al. 2022
    # Not this returns a value from 0 to 1
    return 0.5052*(58.1*MU/N_MU_sim+120**((MU/N_MU_sim)**1.83)) /100