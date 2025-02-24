def Ith_distrib_func(N_mu, N_mu_tot): #Obtained from literature data
    '''
    Adapted from Caillet et al. 2022
    '''
    Ith=3.85*10**-9*9.1**((N_mu/N_mu_tot)**1.1831)#2.09*10**-9*16**((MN/400)**1.5725)
    return Ith