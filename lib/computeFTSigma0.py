'''
Calculate the relative maximum isometric stresses to obtain a given whole muscle maximum isometric stress

Assume ratio between slow and fast fibre specific tension given by 
'''
import numpy as np 

# Generate a Butterworth low-pass filter
def computeFTSigma0(sigma_0, alpha_s): 
    sigma_0_slow = sigma_0 / (1.27 - 0.27 * alpha_s)
    sigma_0_fast = 1.27 * sigma_0_slow
    return sigma_0_slow, sigma_0_fast
