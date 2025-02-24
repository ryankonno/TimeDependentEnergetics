def importExpData(datapath):
    '''
    Function adapted from Arnault Caillet 
    '''
    import scipy.io
    import numpy as np

    mat = scipy.io.loadmat(datapath) 

    #Extracting relevant data    
    for key, value in mat.items(): 
        if key=='MUPulses':
            disch_times_raw=np.array((value))[0]
        if key=='ref_signal':
            Force=np.array((value))[0]
        if key=='MVC': 
            MVC = np.array((value))[0]    

    # Number of recorded MNs
    Nb_MN=len(disch_times_raw) 

    # Ordering the spike trains from earliest to latest first discharge time
    disch_times_disorganised=np.empty((Nb_MN,), dtype=object) 
    first_disch=np.ones(Nb_MN) #storing the first discharge times, helping in raking the data

    # first, reshaping the spike train data, and storing the first discharge times of each spike train    
    for i in range (Nb_MN): 
        disch_times_disorganised[i]=disch_times_raw[i][0].astype(object)
        first_disch[i]=disch_times_disorganised[i][0] #adding the recruitment time
        
    # then, going through the array of recruitment times, and ranking each index of first_disch to sort the data
    order = first_disch.argsort()
    ranks = order.argsort()        
    disch_times=np.empty((Nb_MN,), dtype=object) 
    for i in range (Nb_MN): 
        j=np.argwhere(ranks==i)[0][0]
        disch_times[i]=disch_times_disorganised[j]        

    # Export data 
    dataout = {
        'N_MU': Nb_MN,
        'force': Force,
        'DT': disch_times,
        'MVC': MVC
    }
                        
    return dataout