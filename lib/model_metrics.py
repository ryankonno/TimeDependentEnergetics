#######################################################
# Compute r2 
import numpy as np
def r2_score(y_pred, y_exp):
    # Use corr coeff matrix
    # corr_matrix = np.corrcoef(y_exp, y_pred)
    # # print(corr_matrix)
    # corr = corr_matrix[0,1]
    # r2 = corr**2 
    # print(f'R^2 (by corcoeff) = {r2}')

    ss_res = np.sum((y_pred - y_exp)**2)
    print(f'ss_res = {ss_res}')
    ss_tot = np.sum((y_exp-np.mean(y_exp))**2)
    print(f'ss_tot = {ss_tot}')
    r2 = 1 - (ss_res / ss_tot)
    # print(f'R^2 (by defn) = {r2}')

    return r2 

# Compute mean square error 
def mse_calc(y_pred, y_exp): 
    return np.sqrt(np.sum( (y_pred - y_exp)**2 )) / np.size(y_pred,0)

# Compute relative error (L2 norm)
def l2_err(y_pred, y_exp): 
    return np.linalg.norm(y_pred - y_exp) / np.linalg.norm(y_pred)