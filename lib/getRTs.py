def getRTs(DT, force):
    '''
    Function to get the recruitment thresholds

    Note that this function is written with respect to force, 
    but could also be used to get the recruitment with respect to the common drive

    Everything needs to be at the same sample frequency
    '''
    import numpy as np

    # Initialize the vector of recruitment thresholds
    RTs = np.zeros(np.size(DT))

    for i in range(np.size(DT)):
        RTs[i] = force[int(DT[i][0])]

    return RTs