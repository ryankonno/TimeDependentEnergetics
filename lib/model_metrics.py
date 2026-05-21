import numpy as np

# Compute r2 
def r2_score(y_pred, y_exp):
    ss_res = np.sum((y_pred - y_exp)**2)
    ss_tot = np.sum((y_exp-np.mean(y_exp))**2)
    r2 = 1 - (ss_res / ss_tot)
    return r2 

# Compute normalised root mean square error 
def nrmse_calc(y_pred, y_exp): 
    return np.sqrt(np.sum( (y_pred - y_exp)**2 ) / np.size(y_pred,0)) / np.mean(y_exp)

# Compute relative error (L2 norm)
def l2_err(y_pred, y_exp): 
    return np.linalg.norm(y_pred - y_exp) / np.linalg.norm(y_pred)