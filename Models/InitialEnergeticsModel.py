'''
Initial energetics model code.

Includes the model from Konno et al., 2025

Note the current version here ignores lengthening heat rates... This will need to be updated for experiments including lengthening (see commented lines)

Ryan Konno
r.konno@uq.edu.au
The University of Queensland 
'''
################################################################################
# Import
import numpy as np

# Import the force-length relationship for the mechanical model
from Models.MechanicsModel import MechModel

################################################################################
class EnergeticsModel():
    def __init__(self):
        return None
    ################################################################################
    def solveInitialEnergetics(
        self, t, ca_vec, catn_vec, params,
        e_m=None, dedt_m=None, force=None, mech_model=None
    ):
        '''
        Solve the energetics fo the 

        Input :
            t: time 
            dcadt_vec: rate of Ca transport into the cytoplasm 
            catn_vec: concentration of bound CaTn complex (assume proportional to force)
            params: muscle parameters 
        Output:
            q_cat: Ca transport heat rate (F0l0/s)
            q_cxb: Cross-bridge cycling heat rate (F0l0/s)
            q_sl: Shortening heat rate (F0l0/s)
            w: Work rate (F0l0/s)
        '''

        # Define constants associated with the activation and mainenance processes 
        r_cat = params['r_cat']
        r_cxb = params['r_cxb']
        cxb_scale = params.get('cxb_scale', 1)

        # Compute rate of ca transport for later calculation
        # Compute time differences, prepend first value to match dimensions
        dt = np.diff(t, prepend=t[0])
        # Replace zeros in time differences with a small epsilon
        dt[dt == 0] = 1e-10
        # Safely compute derivative
        dcadt_vec = np.diff(ca_vec, prepend=ca_vec[0]) / dt

        # Compute heat of Ca transport (valid if ca_vec is decreasing)
        q_cat = -r_cat * dcadt_vec * (dcadt_vec < 0) # No scaling based on Ca concentration 

        # Compute cross-bridge energetics (maintenance)
        # This will be scaled based on catn_vec
        q_cxb_0 = r_cxb * (catn_vec ** cxb_scale)

        # Length dependent parts of the code
        if e_m is None or not e_m.any(): 
            # No length dependence
            q_cxb = q_cxb_0
            
            return q_cat, q_cxb # Reported in F0l0/s
        else:
            # Assume that we want to compute the length dependent aspects
            # Maintenance 
            q_cxb = q_cxb_0 * mech_model.F_la(e_m) 

            # Shortening-lengthening heat rate
            # q_sl = (- params['r_sl'] * catn_vec * mech_model.F_la(e_m) *
            #         dedt_m * (dedt_m < 0) +
            #         dedt_m * force * (dedt_m <= 0))
            # Ignore lengthening heat (set to zero)
            q_sl = (
                - params['r_sl'] * catn_vec * mech_model.F_la(e_m) *
                dedt_m * (dedt_m < 0)
            )

            # Work
            w = -force * dedt_m * (dedt_m < 0)

            return q_cat, q_cxb, q_sl, w # F0l0/s

    ################################################################################
    def dHdt_Konno2025(self, act, t, e_ce, dedt_ce, F, r1, r2, params):
        '''
        Computes heat rates using the model from Konno et al., 2025

        Inputs: 
            act: muscle activation (unitless)
            t: time vector (s) 
            e_ce: CE strain (unitless)
            dedt_ce: CE strain rate (1/s) 
            F: Muscle force (unitless) 
            r1: Activation and maintenance heat parameter (1/s) 
            r2: Shortening heat parameter (unitless) 
            params: Muscle parameters 
        Outputs: 
            Energy_rate_data: Dictionary of energetic rates (W) 
            Energy_data: Dictionary of energy values (J)
        '''

        # Define parameters
        l_0 = params[params['muscle']]['l_0']
        F_0 = params[params['muscle']]['F_0']

        mech_model = MechModel(
            params[params['muscle']]['l_0'],
            params[params['muscle']]['dedt_ce_max'],
            params[params['muscle']]['kappa'],
            params['k_see']
        )

        # s, Assume constant spacing
        dt = t[1] - t[0]

        # Compute the total maintenance heat rate, Output in 1/s
        def dQmdt_total(t, act, e_m, dedt_m, F):
            
            # Maintenance heat rate
            def dQmdt(t, act, e_m, dedt_m, F):
                # Adjusted for +ive dedt_m during lengthening
                v_ce_g0 = 0.3 + 0.7 * np.exp(-8 * dedt_m)
                return (
                    r1 * ((dedt_m < 0) +
                          v_ce_g0 * (dedt_m >= 0))
                )

            return (
                act * dQmdt(t, act, e_m, dedt_m, F) *
                (0.3 + 0.7 * mech_model.F_la(e_m))
            )

        # Compute the total shortening lengthening heat rate, Output in 1/s
        def dQsldt_total(t, act, e_m, dedt_m, F):
            
            # Previous definition (commented out)
            # # Shortening and lengthening heat rate
            # def dQsldt(act, e_m, dedt_ce, F):
            #     return (- r2 * dedt_ce * (dedt_ce < 0) +
            #             - F * (-dedt_ce) * (dedt_ce >= 0))
            #     # Assumes all external work lost as heat
            # return act * mech_model.F_la(e_m) * dQsldt()

            # All work converted to heat (independent of Fla relations)
            def dQsldt(act, e_m, dedt_ce, F):
                # Assumes all external work lost as heat
                return (
                    - r2 * dedt_ce * mech_model.F_la(e_m) *
                    (dedt_ce < 0) 
                    # + F * (-dedt_ce) * (dedt_ce >= 0) # Ignoring lengthening heat rates
                )
            return act * dQsldt(act, e_m, dedt_m, F)

        # Compute the total heat rate, Output in 1/s
        def dQdt(t, act, e_m, dedt_m, F):
            return dQmdt_total(t, act, e_m, dedt_m, F) + dQsldt_total(t, act, e_m, dedt_m, F)

        # Calculate the work done by the contractile unit
        def W(dedt_ce, F):
            return -dedt_ce * F  * (dedt_ce < 0)# Output in 1/s

        # Compute the energy rates and scale to W
        Q_tot    = dQdt(t, act, e_ce, dedt_ce, F) * F_0 * l_0 # W
        Q_m_tot  = dQmdt_total(t, act, e_ce, dedt_ce, F) * F_0 * l_0 # W
        Q_sl_tot = dQsldt_total(t, act, e_ce, dedt_ce, F) * F_0 * l_0 # W
        W_tot    = W(dedt_ce,F) * F_0 * l_0 # W
        E_init   = Q_tot + W_tot # W
        Q_rec    = 1 * E_init # W, NOTE: set recovery ratio to 1
        dEdt = E_init + Q_rec# W

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

        return Energy_rate_data, Energy_data # W, J
