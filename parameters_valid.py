'''
Parameters to use in the validation codes 
'''
# Parameters
params = {
    # General muscle parameters
    'rho0':  1e6, # g/m^3, Density of muscle

        # Mouse data 
        'SOL': {

            # Excitation-activation parameters
            'Tau_1': 0.038,  # s, Close et al., 1967
            'Tau_2': 0.055,  # s, Close et al., 1967
            "K": 0.25, # 
            "n": 1.99, # Hill coefficient for act mdoel

            # Mechanical parameters 
            'F_0': 0, # N, Barclay and Weber 2004
            'l_0': 11e-3, # m, Barclay and Weber 2004
            'mass': 4.1e-3, # g, Barclay and Weber 2004
            'max_iso_stress': 2.37e5, # N/m^2, Barclay 1996
            'dedt_ce_max': 6,  # s^-1, Barclay 2010
            'kappa': 0.18,
            
            # Initial energetics model 
            'r_cxb':   0.2584, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cat': 0.03540, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'cxb_scale':  0.5166, # unitless, cxb scale factor
            'r_sl':  0.1258, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

            # Bioenergetics parameters 
            'c_c_tot': 29.5, # mM, Kushmerick et al. 1992 
            'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992 
            'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992 
            # Optimised values 
            'V_max_oxphos': 2 * 1.9322, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 0.61325, # unitless, # original
            'r_rec': 0.5 * 0.16730e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 1, # Scaling factor for metabolic rates at rest    
            # # Corrected values 
            # 'V_max_oxphos': 1.9322, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            # 'K_adp': 0.058, # mM,
            # 'nh': 0.61325, # unitless, # original
            # 'r_rec': 0.16730e6, # J / mol, Obtained from efficiency calculation 
            # 'gamma': 1, # Scaling factor for metabolic rates at rest  

            # Other model parameters
            # Konno et al., 2025 model parameters 
            'r1': 0.6177,
            'r2': 0.2342,

        }, 
        'EDL': {

            # Excitation-activation parameters
            'Tau_1': 0.011,  # requested
            'Tau_2': 0.011,  # BH 2003, fibre bundle data
            "K": 0.45,
            "n": 2.89, # Hill coefficient for activation model

            # Mechanical parameters
            'F_0': 0, # N,
            'l_0': 8.9e-3, # m,
            'mass': 3.9e-3, # g,
            'max_iso_stress': 3.01e5, # N/m^2, B1996
            'dedt_ce_max': 11, # s^-1, Barclay 2010
            'kappa': 0.29,

            # Initial energetics model
            'r_cxb':  0.761209, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'r_cat': 0.0216, # F0l0/s, Maximum heat rate of isometric contraction (slow-type fibre)
            'cxb_scale':  0.23276, # unitless, cxb scale factor
            'r_sl':  0.105056, # W/F_0/l_0, Maximum shortening heat rate (slow-type fibre)

            # Bioenergetics parameters
            'c_c_tot': 29.5, # mM, Kushmerick et al. 1992
            'c_atp_0': 5.3, # mM,  Kushmerick et al. 1992
            'c_pcr_0': 21.1, # mM,  Kushmerick et al. 1992
            # Optimised values 
            'V_max_oxphos': 2 * 1.9322, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            'K_adp': 0.058, # mM,
            'nh': 0.61325, # unitless, # original
            'r_rec': 0.5 * 0.16730e6, # J / mol, Obtained from efficiency calculation 
            'gamma': 1, # Scaling factor for metabolic rates at rest    
            # # Corrected values 
            # 'V_max_oxphos': 1.9322, # mM/s, Assume 2x recovery rate at 35 compared to 20 degrees
            # 'K_adp': 0.058, # mM,
            # 'nh': 0.61325, # unitless, # original
            # 'r_rec': 0.16730e6, # J / mol, Obtained from efficiency calculation 
            # 'gamma': 1, # Scaling factor for metabolic rates at rest  

            # Other model parameters
            # Konno et al., 2025 model parameters
            'r1': 2.7919,
            'r2': 0.697,

        },

        # Other bioenergetics parameters
        # Assume constant across all species and muscle fibre-types
        'V_ck_f': 100,# 100, # mM/s, Kushmerick 1998
        'K_b': 1.11, #mM, MacFarland 1994
        'K_ia': 0.135, # mM, MacFarland 1994
        'K_eq': 1.77e2, # Assuming a pH of 7, Lawson 1979
        'K_iq': 3.5, # mM, MacFarland 1994
        'K_ib': 3.9, # mM, MacFarland 1994
        'K_p': 3.8, # mM, MacFarland 1994
        'Gatp': 60e3, # J/mol, Free energy of ATP (Barclay 2019)

        # Mechanical model SEE stiffness (unused)
        'k_see': 0, 
}