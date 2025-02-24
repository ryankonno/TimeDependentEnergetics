def importExpDataMulti(datapath):
    '''
    This code has been designed to handle multiple contractions 

    Function adapted from Arnault Caillet 
    '''
    import scipy.io
    import numpy as np

    mat = scipy.io.loadmat(datapath) 

    # First, lets attempt to handle some nested cells 
    for key, value in mat.items(): 
        if key == 'MUPulses': 
            # Define the broad structure as an np array
            disch_times_raw_unform = np.array((value))[0]
            disch_times_raw = np.empty_like(disch_times_raw_unform)
            disch_times = np.empty_like(disch_times_raw_unform) # This is the final output (formatteds and sorted)

            disch_times_raw[0] = np.empty_like(disch_times_raw_unform[0])

            # Define a variable to store the number of MUs
            N_mu = np.zeros((5,))
            
            # Now we need to reformat so that we do not have so many nested arrays
            for i in range(5):
                # First initialize the vector to be an empty array with the right size 
                disch_times_raw[i] = np.empty_like(disch_times_raw_unform[i])
                
                # Now loop over the MUs 
                N_mu[i] = len(disch_times_raw_unform[i])
                for j in range(int(N_mu[i])):

                    # Define a temporary array with the correct formatting 
                    tmp_array = [np.array((disch_times_raw_unform[i][j][0][0]))]

                    # Add the temporary array to the larger vector
                    disch_times_raw[i][j] = tmp_array
                # print(disch_times_raw[i][j][0])

                ##########
                # Now we need to format the raw signals based on the discharge times 
                # # Ordering the spike trains from earliest to latest first discharge time
                disch_times_disorganised= np.empty_like(disch_times_raw[i])
                first_disch=np.ones((int(N_mu[i]), )) #storing the first discharge times, helping in raking the data

                # first, reshaping the spike train data, and storing the first discharge times of each spike train    
                for k in range(int(N_mu[i])): 
                    # Add the current MU to the disorganized array 
                    tmp_array = [np.array((disch_times_raw[i][k]))]
                    disch_times_disorganised[k] = tmp_array
                    
                    # Get the first discharge time 
                    first_disch[k] = disch_times_disorganised[k][0][0][0] #adding the recruitment time
                    
                # then, going through the array of recruitment times, and ranking each index of first_disch to sort the data
                order = first_disch.argsort()
                ranks = order.argsort()        
                disch_times[i]=np.empty((int(N_mu[i]),), dtype=object)
                for k in range(int(N_mu[i])): 
                    l=np.argwhere(ranks==k)[0][0]
                    disch_times[i][k] = disch_times_disorganised[l][0][0]


        # Handle the torque trace
        if key == 'ref_signal':
            # Define the broad structure as an np array
            ref_signal_unform = np.array((value))[0]

            ref_signal = np.empty_like(ref_signal_unform)

            # disch_times_raw[0] = np.empty_like(disch_times_raw_unform[0])
            
            # Now we need to reformat so that we do not have so many nested arrays
            for i in range(5):
                # First initialize the vector to be an empty array with the right size 
                ref_signal[i] = np.empty_like(ref_signal_unform[i])
                
                # Define a temporary array with the correct formatting 
                tmp_array = [np.array((ref_signal_unform[i][0]))]

                # Add the temporary array to the larger vector
                ref_signal[i] = tmp_array
                
            # print(ref_signal[0)

        if key=='MVC': 
            MVC = np.array((value))[0]

                    

    # print(disch_times_raw)
    # Export data 
    dataout = {
        'N_MU': N_mu,
        'force': ref_signal,
        'DT': disch_times,
        'MVC': MVC
    }
                   
            

    return dataout











    # #Extracting relevant data    
    # for key, value in mat.items(): 
    #     if key=='MUPulses':
    #         disch_times_raw=np.array((value))[0]
    #     if key=='ref_signal':
    #         Force=np.array((value))[0]
    #     if key=='MVC': 
    #         MVC = np.array((value))[0]    

    # # Number of recorded MNs
    # Nb_MN=len(disch_times_raw) 

    # # Ordering the spike trains from earliest to latest first discharge time
    # disch_times_disorganised=np.empty((Nb_MN,), dtype=object) 
    # first_disch=np.ones(Nb_MN) #storing the first discharge times, helping in raking the data

    # # first, reshaping the spike train data, and storing the first discharge times of each spike train    
    # for i in range (Nb_MN): 
    #     disch_times_disorganised[i]=disch_times_raw[i][0].astype(object)
    #     first_disch[i]=disch_times_disorganised[i][0] #adding the recruitment time
        
    # # then, going through the array of recruitment times, and ranking each index of first_disch to sort the data
    # order = first_disch.argsort()
    # ranks = order.argsort()        
    # disch_times=np.empty((Nb_MN,), dtype=object) 
    # for i in range (Nb_MN): 
    #     j=np.argwhere(ranks==i)[0][0]
    #     disch_times[i]=disch_times_disorganised[j]        

    # # Export data 
    # dataout = {
    #     'N_MU': Nb_MN,
    #     'force': Force,
    #     'DT': disch_times,
    #     'MVC': MVC
    # }
                        
    # return dataout




def importActDataMulti(datapath):
    import numpy as np 
    import scipy.io 

    # Import the data     
    mat = scipy.io.loadmat(datapath) 

    # First, lets attempt to handle some nested cells 
    for key, value in mat.items(): 
        if key == 'act_max':
            act_max = np.array((value))[0]

        if key == 'act_max_mvc':
            act_mvc_max = np.array((value))[0]

    data_out ={
        'act_max': act_max, 
        'act_mvc_max': act_mvc_max
    }

    return data_out