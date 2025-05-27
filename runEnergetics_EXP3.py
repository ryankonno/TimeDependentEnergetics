'''
Code to compare the energetic cost of increasing the amount of force produced by a given MU by x%. 

Options to increase force include firing the MU at a faster rate OR recruiting another MU.

Ryan Konno
School of Biomedical Sciences 
The University of Queensland
'''

# Import 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
sys.path.append('./')

# Parameters 
params = {
    # Simulation specific 
    'dF_factor': 0.1, # unitless, factor to increase the force by (dF_factor * F_mean)

    't_start': 0, # s
    't_end': 100, # s
    # 'N_contractions':50,
    'fsamp': 2048, # Hz
    'N_t': 100000, # int, Number of time steps (activation model)

    'r_stim_1': 20, # Hz, Stimulation frequency for the smaller MU

    'N_MU_sim': 1, # int, Number of MUs to stimulate    

    # MU Size properties (can be changed, these are just approximate values)
    # NOTE: These were estimated based on model estimates given RT experimentally
    #       can be changed....
    'sigma_0': 2e5, # Pa, Maximum isometric stress
    'mu_csa': 0.02e-5, # m^2, motor unit CSA
    'F_0': 2e5 * 0.02e-5, # N, sigma_0 * CSA 
    'l_0': 6.83e-2, # m, resting fascicle length
    'mass': 0.0286, # g, weight of on MU (based on fraction of total muscle CSA and whole muscle mass of 143.1g)

    # Activation model parameters
    'Activation_model': {
        # Time constant for excitation-activation dynamics 
        # Slow properties 
        "Tau_1": 0.0422, # Assume constant value from MCL (2023)
        # "Tau_2": 0.256, # Scaling based on MCL (2023)
        "Tau_2": 0.1,
        # Fast properties 
        # "Tau_1": 0.0422, # Very little change between fibre type - assume constant (BH, 2003)
        # "Tau_2": 0.256/2, # Decay constant for fast twitch Fibres assuming 1/2 rate (Baylor and Hollingworth, 2003)

        "K": 0.1025,
        "n": 3, # Hill coefficient
    },

    # Energetics parameters 
    'Energetics_model':{
        # 'nu': 1.72,

        # 'H_PCr': 36e3, # J/mol, Enthalpy of pcr

        'r_am': 0.6177, # W/F_0/l_0, Maximum heat rate of isometric contraction (slow-type fibre)

                # Define parameters for the ODE 
        # .... In this code they are used as initial values for the optimization
        'k_recpcr': 0.0214, # 1/s
        'k_m_1':  0.0367, # 1/s

        # Energetic constant to predict energetic rates
        # 'r_r': 522e-3, # J (umol)^-1, Calculated based on the glycogen and atp enthalpys
        'r_r': 0.04489659, # J (umol)^-1, Optimized to experimental data (Phillips et al. 1993)
    }
}

### Use a loop to try different dF_factor values 

dF_factor_vec = np.array((0.05, 0.1)) 
# dF_factor_vec = np.array((0.1, 0.25, 0.5, 0.75, 1.0))


for exp_idx, dF_factor in enumerate(dF_factor_vec):

    print('_______________________________')
    print('_______________________________')
    print(f'Running dF_factor = {dF_factor}')

    ###
    # Set the MU sizes (CSA) 
    mu_csa_vec =  np.array((params['mu_csa'], params['mu_csa'], params['mu_csa']))
    mu_mass_vec = np.array((params['mass'], params['mass'], params['mass']))
    F_0_vec = mu_csa_vec * params['sigma_0']

    # Initialize storage vectors 
    force = np.empty_like(mu_csa_vec,dtype=object)
    catn_vec = np.empty_like(mu_csa_vec,dtype=object)
    ca_vec = np.empty_like(mu_csa_vec,dtype=object)
    force_mean = np.empty_like(mu_csa_vec,dtype=float)
    r_stim_vec = np.empty_like(mu_csa_vec,dtype=float)
    t_fire_idxs = np.empty_like(mu_csa_vec, dtype = object)

    q_tot_vec = np.empty_like(mu_csa_vec, dtype = object)
    q_i_vec = np.empty_like(mu_csa_vec, dtype = object)
    q_r_vec = np.empty_like(mu_csa_vec, dtype = object)

    q_tot_mean = np.empty_like(mu_csa_vec, dtype = float)
    q_i_mean = np.empty_like(mu_csa_vec, dtype = float)
    q_r_mean = np.empty_like(mu_csa_vec, dtype = float)


    ###
    '''
    Functions to compute stimulation times, activations, and energeticss
    '''
    def computeMUStim(r_stim, t_vec_):
        # Now get the firing times and then the indices in t_vec
        period = 1/np.maximum(1,r_stim)
        N_t_fire = params['t_end'] / period
        t_fire_ideal = [i * period for i in range(int(N_t_fire))] # Ideal firing times 
        t_fire_idxs_ = np.searchsorted(t_vec_, t_fire_ideal, side='left') # Get the indices for the first MU
        return t_fire_idxs_

    from Models.MUActivationModel import ActivationModel
    def computeActivation(t_vec_, t_fire_idxs_): 
        # Initialize 
        act_model = ActivationModel(params['Activation_model'], t_vec_, True)

        # Run the Ca dynamics
        stim_vec, ca_vec, catn_vec = act_model.runExcAct(t_fire_idxs_)

        return ca_vec, catn_vec

    from Models.MUEnergeticsModel import EnergeticsModel
    def computeEnergetics(t_vec_, ca_vec_, catn_vec_, F_0_, l_0_, mass_): 
        # Initialize energetics model 
        energy_model = EnergeticsModel()

        # Compute the energetics based on Ca and CaTn 
        q_a, q_m = energy_model.actEnergetics(t_vec_, ca_vec_, catn_vec_, params['Energetics_model']) # W/F_0/l_0

        # Convert from W/F_0/l_0 to W/g 
        s_f = F_0_ * l_0_ / mass_
        q_i_W_g = (q_a + q_m) * s_f

        # Compute the recovery energetic rate using teh bioenergetics model 
        from Models.BioenergeticsModel import Bioenergetics
        model = Bioenergetics(params['Energetics_model']['k_recpcr'], params['Energetics_model']['k_m_1'],t_vec_, q_i_W_g)
        sol = model.solveODE(t_vec_[0], t_vec_[-1], t_vec_) # Solve the IVP 
        PCr, activation = sol.y

        q_r_W_g = model.computeEnergetics(PCr, activation, params['Energetics_model']['r_r']) # W/g, Compute the recovery energetic rate

        # Scale from W/g to W
        q_r = q_r_W_g * mass_
        q_i = q_i_W_g * mass_

        # Compute the total heat rate
        q_tot = q_r + q_i 

        return q_tot, q_i, q_r

    # Set the time vector 
    t_vec = np.linspace(params['t_start'], params['t_end'], params['t_end'] * params['fsamp'])

    # Define the set firing times for the smaller MU
    r_stim_vec[0] = params['r_stim_1']
    print(f'Firing rate for MU: {r_stim_vec[0]}')
    t_fire_idxs[0] = computeMUStim(r_stim_vec[0], t_vec)

    ### 
    '''
    Compute the baseline motor unit simulation
    '''
    # Compute the activation
    ca_vec[0], catn_vec[0] = computeActivation(t_vec, t_fire_idxs[0])

    # Compute the force based on the CSA 
    force[0] = catn_vec[0] * mu_csa_vec[0] * params['sigma_0']

    # Save the mean force 
    force_mean[0] = np.mean(force[0])


    # Compute the energetics 
    q_tot_vec[0], q_i_vec[0], q_r_vec[0] = computeEnergetics(t_vec, ca_vec[0], catn_vec[0], F_0_vec[0], params['l_0'], mu_mass_vec[0])
    q_tot_mean[0], q_i_mean[0], q_r_mean[0] = np.mean(q_tot_vec[0]), np.mean(q_i_vec[0]), np.mean(q_r_vec[0])

    # # Plot Ca dynamics for the first MU
    # fig_cadyn, axs_cadyn = plt.subplots(1, 2, figsize = (12,6), layout = 'constrained')

    # ax = axs_cadyn[0]
    # ax.plot(t_vec, ca_vec[0] )
    # ax.set_xlim((0,4))
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('[Ca2+] (Normalized)')

    # ax = axs_cadyn[1]
    # ax.plot(t_vec, force[0])
    # ax.set_xlim((0,4))
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Force (N)')
    # plt.show() 

    '''
    Determine the new force (F_star) that we want to match
    '''
    F_star = (1 + dF_factor) * force_mean[0]

    print(f'Original force: {force_mean[0]} N')
    print(f'New force (F_star): {F_star} N')
    print('')

    ### 
    # Compute stimulation frequency required to match the mean force between the MUs
    '''
    Assume that the MUs in the loop will be as follows 
    1) MU increased firing rate 
    2) New MU to account for change in force 
    '''
    

    for mu, mu_csa in enumerate(mu_csa_vec[1:]):
        # Adjust mu number 
        mu += 1

        print('__________________________________')
        print(f'Optimizing MU: {mu}')

        # Case 1) 
        if mu == 1: 
            print('    MU1: optimizing firing frequency to match F_star...')
            # In this case we simply optimize the MU firing rate to match F_star
            # Define a function to optimize
            def f_opt(r_stim_mu):

                # Compute the time vector 
                t_fire_idxs_ = computeMUStim(r_stim_mu, t_vec)

                # Compute the activation level 
                _, catn_vec_ = computeActivation(t_vec, t_fire_idxs_)

                # Compute the force 
                force_mean_ = np.mean(catn_vec_ * mu_csa_vec[mu] * params['sigma_0'])

                # Error 
                error = np.abs(F_star - force_mean_) 

                return error 
            
            # Optimize over the stimulation frequency 
            from scipy.optimize import minimize_scalar
            opt_res = minimize_scalar(f_opt, bounds = (1,50))

            # Save the recorded stimulation frequency
            r_stim_vec[mu] = opt_res.x
            print(f'    Optimized firing frequency: {r_stim_vec[mu]}')
            print(f'    Error: {opt_res.fun}')

            # Rerun and save the model results with the optimal stim frequency 
            t_fire_idxs[mu] = computeMUStim(r_stim_vec[mu], t_vec)
            ca_vec[mu], catn_vec[mu] = computeActivation(t_vec, t_fire_idxs[mu])
            force[mu] = catn_vec[mu] * mu_csa_vec[mu] * params['sigma_0']
            force_mean[mu] = np.mean(force[mu])

            print(f'    Mean force = {force_mean[mu]} N')

            # Compute the energetics
            q_tot_vec[mu], q_i_vec[mu], q_r_vec[mu] = computeEnergetics(t_vec, ca_vec[mu], catn_vec[mu], F_0_vec[mu], params['l_0'], mu_mass_vec[mu])
            q_tot_mean[mu], q_i_mean[mu], q_r_mean[mu] = np.mean(q_tot_vec[mu]), np.mean(q_i_vec[mu]), np.mean(q_r_vec[mu])

            print(f'    Mean energy rate = {q_tot_mean[mu]} N')

        # Case 2) 
        elif mu == 2: 
            print('     MU: Recruiting another MU to reach F_star')

            def f_opt(r_stim_mu):

                # Compute the time vector 
                t_fire_idxs_ = computeMUStim(r_stim_mu, t_vec)

                # Compute the activation level 
                _, catn_vec_ = computeActivation(t_vec, t_fire_idxs_)

                # Compute the force 
                force_mean_ = np.mean(catn_vec_ * mu_csa_vec[mu] * params['sigma_0'])

                # Error 
                error = np.abs(F_star - force_mean_ - force_mean[0])

                return error
        
            # Optimize over the stimulation frequency 
            from scipy.optimize import minimize_scalar
            opt_res = minimize_scalar(f_opt, bounds = (1,50))

            # Save the recorded stimulation frequency
            r_stim_vec[mu] = opt_res.x
            print(f'    Optimized firing frequency: {r_stim_vec[mu]}')
            print(f'    Error: {opt_res.fun}')

            # Rerun and save the model results with the optimal stim frequency 
            t_fire_idxs[mu] = computeMUStim(r_stim_vec[mu], t_vec)
            ca_vec[mu], catn_vec[mu] = computeActivation(t_vec, t_fire_idxs[mu])
            force[mu] = catn_vec[mu] * mu_csa_vec[mu] * params['sigma_0'] + force[0]
            force_mean[mu] = np.mean(force[mu])

            print(f'    Mean force = {force_mean[mu]} N')

            # Compute the energetics
            q_tot_vec[mu], q_i_vec[mu], q_r_vec[mu] = computeEnergetics(t_vec, ca_vec[mu], catn_vec[mu], F_0_vec[mu], params['l_0'], mu_mass_vec[mu])
            # Add the energetic cost of the first MU to the total values 
            q_tot_vec[mu] += q_tot_vec[0]
            q_i_vec[mu] += q_i_vec[0]
            q_r_vec[mu] += q_r_vec[0]

            # Compute the means
            q_tot_mean[mu], q_i_mean[mu], q_r_mean[mu] = np.mean(q_tot_vec[mu]), np.mean(q_i_vec[mu]), np.mean(q_r_vec[mu])

            print(f'    Mean energy rate = {q_tot_mean[mu]} N')


    # Define plot for showing the cumulative means
    fig, ax = plt.subplots(figsize = (6,4), layout = 'constrained') 
    ls_opts = (':','-')
    import matplotlib.cm as cmap
    c_map = cmap.viridis(np.linspace(0,1,np.size(mu_csa_vec)))
    for mu in range(np.size(mu_csa_vec)):    
        ax.set_title(f'MU Exp {mu}')
        ax.set_xlabel(f'Time')
        ax.set_ylabel(f'Cumulative mean energy rate (W)')
        ax.plot(t_vec, np.cumsum(q_tot_vec[mu]) / np.arange(1,np.size(q_tot_vec[mu]) + 1), label = r'$q_{tot}$, ' + f'exp = {mu}', color = c_map[mu])
        ax.plot(t_vec, np.cumsum(q_i_vec[mu]) / np.arange(1,len(q_i_vec[mu]) + 1), label = r'$q_{i}$, ' + f'exp = {mu}', ls = ':', color = c_map[mu])
        ax.plot(t_vec, np.cumsum(q_r_vec[mu]) / np.arange(1,len(q_r_vec[mu]) + 1), label = r'$q_{r}$, ' + f'exp = {mu}', ls = '--', color = c_map[mu])
    ax.legend()
    fig.savefig('./Figures/MUEnergyCumSumMean_EXP3_df_' + str(dF_factor) + 'r_0_stim_' + str(r_stim_vec[0] ) + '.pdf')



    # Plot the corresponding results for all of the MUs
    fig, axs = plt.subplots(1, 4, figsize = (12,4), layout = 'constrained') 
    ax = axs[0] 
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('q_{tot} (W)')
    ax = axs[1] 
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('q_{i} (W)')
    ax = axs[2] 
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('q_{r} (W)')

    for mu in range(np.size(mu_csa_vec)):
        ax = axs[0] 
        ax.plot(t_vec, q_tot_vec[mu], label  = 'MU: ' + str(mu)) 
        ax = axs[1] 
        ax.plot(t_vec, q_i_vec[mu], label  = 'MU: ' + str(mu)) 
        ax = axs[2] 
        ax.plot(t_vec, q_r_vec[mu], label  = 'MU: ' + str(mu)) 

    ax.legend()


    ax = axs[3] 
    axis_labels = ('Original', 'Exp 1', 'Exp 2')
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Mean energetic rate (W)')
    ax.plot(axis_labels, q_tot_mean, label = 'q_{tot}', ls = 'None', marker = '.')
    ax.plot(axis_labels, q_i_mean, label = 'q_i', ls = 'None', marker = '.')
    ax.plot(axis_labels, q_r_mean, label = 'q_r', ls = 'None', marker = '.')
    ax.legend()

    fig.savefig('./Figures/MUCompEnergy_EXP3_df_' + str(dF_factor) + 'r_0_stim_' + str(r_stim_vec[0] ) + '.pdf')


    # Plot the force results
    fig, axs = plt.subplots(1, 2,figsize = (8,4), layout = 'constrained') 
    ax = axs[0] 
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    for mu in range(np.size(mu_csa_vec)):
        ax.plot(t_vec, force[mu], label  = 'MU: ' + str(mu)) 

    ax = axs[1] 
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Mean force (N)')
    ax.plot(axis_labels, force_mean, ls = 'None', marker = '.')

    fig.savefig('./Figures/MUCompForce_EXP3_df_' + str(dF_factor) + 'r_0_stim_' + str(r_stim_vec[0] ) + '.pdf')

plt.show()
