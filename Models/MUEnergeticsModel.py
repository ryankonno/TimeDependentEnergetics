'''
Energetics model for a single MU/fibre

Heat rates are computed first using unitless quantities, then scaled to W.

Ryan Konno
The University of Queensland
r.konno@uq.edu.au
'''
################################################################################
# Import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# import matplotlib
# matplotlib.use('agg')

# Import the force-length relationship for the mechanical model 
from Models.MechanicsModel import MechModel
from lib.MUScalingFunctions import muCSADistr, varMUDist

################################################################################
class EnergeticsModel():
    def __init__(self):
        return None
    
    def actEnergetics(self, t, ca_vec, catn_vec, params, e_m = None, dedt_m = None, force = None, params_mech = None):
        '''
        Activation component of the energetics 
            designed for investigating Ca dependent properties of activation model

        Input 
            t: time 
            dcadt_vec: rate of Ca transport into the cytoplasm 
            catn_vec: concentration of bound CaTn complex (assume proportional to force)
        '''

        # Define constants associated with the activation and mainenance processes 
        r_a = 0.4 * params['r_am']
        r_m = 0.6 * params['r_am']

        # Compute rate of ca transport for later calculation 
        dt = np.diff(t, prepend=t[0])  # Compute time differences, prepend the first value to match dimensions
        dt[dt == 0] = 1e-10           # Replace zeros in time differences with a small epsilon
        dcadt_vec = np.diff(ca_vec, prepend=ca_vec[0]) / dt  # Safely compute derivative
                
        # Compute heat of Ca transport
        # Valid if ca_vec is decreasing 

        # NOTE: This equation below assumes that pumping is fully buffered by PCr
        R = 8.314 # J/mol/K
        T = 310 # K, body temp
        F = 96485 # C/mol
        z = 2 # unitless, Charge of Ca ions
        V = 0.05 # V, Membrane potential
        C_cyto = (4000 - 0.05) * ca_vec + 0.05 # uMol, free Ca concentration in cytoplasm, Bakker et al. 2017 TODO correct for CaTn
        C_sr = 4000 - C_cyto # uMol, Ca concentration in SR, Bakker et al. 2017
        e_pot_sr = R * T * np.log(C_cyto / C_sr) + F * z * V # J/mol, Electrochemical potential
        # q_a = 1e-3 * dcadt_vec * params['H_pcr'] / 2 / params['nu_sr'] #TODO: J/g, heat rate of pumping Ca back into SR (Barclay Laukonis 2021) 
        # q_a = e_pot_sr * np.diff(C_cyto, prepend = 0) * 1e-6 * dcadt_vec # Energetic cost NOTE: here we do not use the experimentally measured values for the whole fibre bundle heat rates
        # q_a = e_pot_sr * dcadt_vec * 3999.95 * 1e-6 # Energetic cost NOTE: here we do not use the experimentally measured values for the whole fibre bundle heat rates
        q_a = - r_a * dcadt_vec * (dcadt_vec < 0) # 1/s, Use measured quantity scaled based on Ca released

        # Compute cross-bridge energetics (maintenance)
        # This will be scaled based on catn_vec
        q_m_0 = r_m * catn_vec # 1/s, heat rate of cross-bridge kinetics

        # Length dependent parts of the code
        if e_m is None or not e_m.any(): 
            # No length dependence
            q_m = q_m_0
            
            return q_a, q_m # Reported in 1/s
        else:
            # Assume that we want to compute the length dependent aspects 

            # Import a mechanics model for the force-length and force-velo relations
            from Models.MechanicsModel import MechModel
            mech_model = MechModel(params_mech)

            # Maintenance 
            q_m = q_m_0 * mech_model.F_la(e_m) 

            # Shortening-lengthening heat rate 
            # q_sl = - params['r_sl'] * catn_vec * mech_model.F_la(e_m) * dedt_m * (dedt_m < 0) \
            #             + dedt_m * mech_model.F_la(e_m) * force * (dedt_m <= 0)
            # Ingore lengthening heat (set to zero)
            q_sl = - params['r_sl'] * catn_vec * mech_model.F_la(e_m) * dedt_m * (dedt_m < 0)

            # Work 
            # NOTE: this force here should be the MU force??
            w = force * dedt_m * (dedt_m < 0)


            return q_a, q_m, q_sl, w # Reported in 1/s

    
    def dHdt(self, act, t, e_ce, dedt_ce, F, r1, r2, params):

        # Define parameters
        l_0 = params['l_0']
        F_0 = params['F_0']
        width = params['F_la_width']

        mech_model = MechModel(params)

        dt = t[1]-t[0] # s, Assume constant spacing

        # Compute the total maintenance heat rate, Output in 1/s
        def dQmdt_total(t, act, e_m, dedt_m, F):
            
            # Maintenance heat rate
            def dQmdt(t, act, e_m, dedt_m, F):
                v_ce_g0 = 0.3 + 0.7 * np.exp(-8 * dedt_m) # Adjusted for +ive dedt_m during lengthening
                return r1 * ((dedt_m < 0) + v_ce_g0 * (dedt_m >= 0)) # No Labile (scale factor s_labile)

            return act * dQmdt(t, act, e_m, dedt_m, F) * (0.3 + 0.7 * mech_model.F_la(e_m))

        # Compute the total shortening lengthening heat rate, Output in 1/s
        def dQsldt_total(t, act, e_m, dedt_m, F):
            
            # Previous definition
            # # Shortening and lengthening heat rate
            # def dQsldt(act, e_m, dedt_ce, F):
            #     return - r2 * dedt_ce * (dedt_ce < 0) + - F * (-dedt_ce) * (dedt_ce >=0) # Assumes all external work lost as heat
            # return act * mech_model.F_la(e_m) * dQsldt(act, e_m, dedt_m, F)

            # All work converted to heat (independent fo Fla relations)
            def dQsldt(act, e_m, dedt_ce, F):
                return - r2 * dedt_ce * mech_model.F_la(e_m) * (dedt_ce < 0) + F * (-dedt_ce) * (dedt_ce >=0) # Assumes all external work lost as heat
            return act * dQsldt(act, e_m, dedt_m, F)

        # Compute the total heat rate, Output in 1/s
        def dQdt(t, act, e_m, dedt_m, F):
            return dQmdt_total(t, act, e_m, dedt_m, F) + dQsldt_total(t, act, e_m, dedt_m, F)

        # Calculate the work done by the contractile unit
        def W(dedt_ce, F):
            return -dedt_ce * F # Output in 1/s

        # Compute the energy rates and scale to W
        Q_tot    = dQdt(t, act, e_ce, dedt_ce, F) * F_0 * l_0 # W
        Q_m_tot  = dQmdt_total(t, act, e_ce, dedt_ce, F) * F_0 * l_0 # W
        Q_sl_tot = dQsldt_total(t, act, e_ce, dedt_ce, F) * F_0 * l_0 # W
        W_tot    = W(dedt_ce,F) * F_0 * l_0 # W
        E_init   = Q_tot + W_tot # W
        Q_rec    = params['energetics_model']['recovery_ratio'] * E_init # W
        dEdt = E_init + Q_rec# W
        # print(f'F0 = {F_0}')
        # print(f'l_0 = {l_0}')
        # print(f'dedt_ce = {max(dedt_ce)}')
        # print(f'F = {max(F)}')
        # print(f'Peak work = {F_0 * l_0 * max(dedt_ce * F)}')

        # Integrate to get the energy in J
        Sum_QM   = np.cumsum(Q_m_tot * dt)
        Sum_QSL  = np.cumsum(Q_sl_tot * dt)
        Sum_Qtot = np.cumsum(Q_tot * dt)
        Sum_Wtot = np.cumsum(W_tot * dt)
        Sum_E    = np.cumsum(dEdt * dt)
        Sum_Qrec    = np.cumsum(Q_rec * dt)

        # Store data
        Energy_rate_data = {
            'dEdt': dEdt, 
            'dQdt': Q_tot, 
            'dQ_mdt': Q_m_tot, 
            'dQ_sldt': Q_sl_tot,
            'dWdt': W_tot, 
            'Q_rec': Q_rec,
        }
        Energy_data = {
            'Q_m': Sum_QM,
            'Q_sl': Sum_QSL,
            'Q_tot': Sum_Qtot,
            'W_tot': Sum_Wtot,
            'E': Sum_E, 
            'Q_rec': Sum_Qrec
        }

        return Energy_rate_data, Energy_data
