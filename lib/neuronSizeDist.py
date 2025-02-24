''' 
Function for the distribution of MU sizes 

Ryan Konno
'''
from parameters import params as params

def neuronSizeDist(n_mus, c1, c2, c3):
    '''
    Assumes we are fitting to three constants in the power relationship
    '''
    return c1 * 1e-7 * c3**(((n_mus + 1) / params['N_MU_sim'])**(c2 * 0.4))



# def muSizeDist(n_mus, c1, c2):
#     '''
#     Function to define the full MU size distribution
#     '''
#     return c1 * 1e-7 * 2.4**(((n_mus + 1) / params['N_MU_sim'])**(c2 * 0.4))
