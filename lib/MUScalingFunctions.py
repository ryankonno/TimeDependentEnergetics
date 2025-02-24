'''
Functions used for scaling parameters as a result of changes in fibre and MU tyep
'''
# Import statements
import numpy as np

# Define the mu CSA distribution (loose with CSA vs PCSA here)
def muCSADistr(mu_num, params): 

    # Assume 100 fold difference between smallest and larges units 
    diff = params['force_ratio']

    # Lets assume the area of the muscle is given by 
    A_m = params['muscle_pcsa'] # m^2, TA (Handsfield et al. 2014) 

    # Assume an exponential distribution given by 
    exp_dist = np.exp(np.log(diff)/params['N_MU_sim']*mu_num)

    # Compute the sum 
    sum_dist = sum(exp_dist)

    # The MU distribution is then given by 
    A_mu_dist = A_m / sum_dist * exp_dist

    return A_mu_dist

# First one to compute the MU pcsa frac (normalized)
def muCSADistrNORM(mu_num, params): 
    # Assume 100 fold difference between smallest and larges units 
    diff = params['force_ratio']

    # Assume an exponential distribution given by 
    exp_dist = np.exp( np.log(diff) / params['N_MU_sim'] *mu_num)

    # Compute the sum to normalize
    sum_dist = sum(exp_dist)

    return exp_dist / sum_dist

# define the linear function 

# Define the dist of fast fibre fractions
def alphafDistr(mu_num, c_1 = 0, c_2 = 0, function = 'lin'): 

    # P = N_mu / diff * (N_mu * np.exp(diff) - alpha_f_tot * N_mu * np.exp(diff/alpha_f_tot) \
    #                         - N_mu / diff * (np.exp(diff) - np.exp(diff * alpha_f_tot)) )
    # Q = N_mu / diff * (np.exp(diff) - np.exp(diff * alpha_f_tot))
    # C =  sum(np.exp( np.log(diff) / N_mu *mu_num))

    # c_1 = (alpha_f_tot * (C) - P / N_mu) / (Q - P / N_mu) 
    # c_2 = (1 - c_1) / N_mu 

    # print(f'c_1 = {c_1}')
    # print(f'c_2 = {c_2}')
    # print(f'P = {P}')
    # print(f'Q = {Q}')
    # print(f'C = {C}')

    if function == 'exp': 
        # print('Using exp dist')
        result = c_1 * np.exp(c_2 * mu_num)
        
    else: 
        # Compute x-intercept 
        if c_2 == 0:
            x_0 = 1000
        else: 
            x_0 = - c_1 / c_2
        result = 0 * (mu_num < x_0) + (c_1 + c_2 * mu_num) * (mu_num >=  x_0)

    return result

# Now lets look a the transition of a variable assuming a sigmoidal transition between the values
def varMUDist(mu_num, fibre_trans_point, var_min, var_max, params): 
    '''
    slope indicates how fast the transition is between var_min and var_max 

    TODO: add constant scaling
    '''
    if params['param_scale'] == 'sig': 
        # print('Using sig dist...')
        return 1/(1+np.exp(- params['sig_smooth'] * (mu_num - fibre_trans_point))) * (var_max - var_min) + var_min
    
    # @Ryan: Exp and linear function have been adapted to give values based on weighted averages int(alpha_f_dist *norm(Csa_dist)) = alpha_f
    elif params['param_scale'] == 'exp': 
        N_plot = 1000 # Number of points to iterate over
        c_2_range = np.linspace(0,0.1, N_plot)
        mus = np.arange(params['N_MU_sim'])
        sum_vals = np.array([np.sum(alphafDistr(mus, np.exp(-c_2 * params['N_MU_sim']), c_2, 'exp') * muCSADistrNORM(mus, params)) for c_2 in c_2_range])
        c_2 = c_2_range[np.argmin(np.abs(sum_vals - (1 - params['alpha_s'])))] # Get the value that minimizes the error in fibre-type predictions
        c_1 = np.exp(-c_2 * params['N_MU_sim'])# Compute c_1 based on the exponential value at N_mu

        alpha_f_dist = c_1 * np.exp(c_2 * mu_num) # Compute the fractions of slow and fast fibres throughout the mu pool
        

        param_values = var_min * (1 - alpha_f_dist)  + var_max * alpha_f_dist
        return param_values
    
    elif params['param_scale'] == 'lin': 
        # print('Using lin dist...')
        N_plot = 1000 # Number of points to iterate over
        c_2_range = np.linspace(0,0.1, N_plot)
        mus = np.arange(params['N_MU_sim'])
        sum_vals = np.array([np.sum(alphafDistr(mus, 1 - c_2 * params['N_MU_sim'], c_2, 'lin') * muCSADistrNORM(mus, params)) for c_2 in c_2_range])
        c_2 = c_2_range[np.argmin(np.abs(sum_vals - (1 - params['alpha_s'])))] # Get the value that minimizes the error in fibre-type predictions
        c_1 = 1 - c_2 * params['N_MU_sim'] # Compute c_1 based on the exponential value at N_mu
        x_0 = -c_1 / c_2

        alpha_f_dist = alphafDistr(mu_num, c_1, c_2, 'lin') # Compute the fractions of slow and fast fibres throughout the mu pool

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(alpha_f_dist)
        # plt.show()

        param_values = var_min * (1 - alpha_f_dist)  + var_max * alpha_f_dist
        return param_values
    
    elif params['param_scale'] == 'const': 
        return (var_min *  params['alpha_s']  + var_max * (1-params['alpha_s'])) * np.ones_like(mu_num)
    
    else:
        print('Incorrect fibre-type distribution!!')
        return None