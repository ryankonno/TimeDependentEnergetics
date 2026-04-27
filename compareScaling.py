'''
Script to analyse scaling of experimental data from ME1988 and B2012 
'''
import numpy as np 

# ME study 
data = {
    'ME1988': 
    {
        'H': 25, # mJ/g
        'm': 6.75e-3, #g 
        'sigma_0': 17.7, # mN/mm^-2 
        'l_0': 7.5e-3, # m
        'd': 1.05, # mm
    },
    'B2012': 
    {
        'H': 4, # mJ/g 
        'm': 4.6e-3, # g 
        'l_0': 11.8e-3, # m 
        'sigma_0': 28.8, # kPa, twitch
    }, 
    'rho': 1e6, #kg/m^3
}

# Compute the scales heat for ME1988
F_0 = (data['ME1988']['d'] / 2)**2 * np.pi * data['ME1988']['sigma_0']  * 1e-3 # N  
print(f'F_0 = {F_0}')
heat_me1988 = data['ME1988']['H'] * data['ME1988']['m'] / F_0 / data['ME1988']['l_0'] 
print(f'Heat ME1988: {heat_me1988}')

# Compute the heat from the B2012 study 
F_0 = data['B2012']['m'] * 1e3/ data['rho'] / data['B2012']['l_0'] * data['B2012']['sigma_0']
print(f'F_0 = {F_0}')
heat_b2012 = data['B2012']['H'] * data['B2012']['m']  / F_0 / data['B2012']['l_0']
print(f'Heat b2012: {heat_b2012}')
