'''
Mechanical model

This model is implemented to compute the activation levels of a muscle given a muscle force as input.
Intrinsic properties of the model are based on Dick et al. 2017, and parameters are modified for the
given experiment.

Author: Ryan Konno, University of Queensland
		r.konno@uq.edu.au
'''
################################################################################
# Import
import numpy as np
import matplotlib.pyplot as plt

class MechModel():
	def __init__(self, params):
		'''
		Take a parameter set as input to the model
		'''

		self.params = params 

		# Import commonly used parameters throughout the model 

		self.ode_max_step = params['ode_max_step'] 	# Maximum step size for the ode solver

		self.l_0 = params['l_0']					# Optimal fibre length
		self.l_m = params['l_m']					# Muscle length
		self.l_mtu = params['l_mtu'] 				# MTU length
		self.l_t = self.l_mtu - self.l_m			# Tendon length
		

		self.width = params['F_la_width'] 			# Width of active force-length relationship

		self.dedt_ce_max = params['dedt_ce_max']
		self.kappa = params['kappa']

		self.k_see = params['k_see']			 	# SEE component stiffness coefficient (linear constant or nonlinear parameter)
		
		
		return None

	###############################################################
    #		INTRINSIC MUSCLE PROPERTIES
	# Define force-length and force-velocity properties for the muscle
	# From Dick et al. 2017
	def F_la(self, e_m):
		skew = 0.6
		round = 2.3
		return np.exp(-np.abs((e_m**skew -1)/self.width)**round)

	def F_va(self, dedt_ce):
		# dedt_ce_max_ = self.getdedt_ce_max()
		return (1+dedt_ce/self.dedt_ce_max)/(1-dedt_ce/self.dedt_ce_max/self.kappa) * (dedt_ce < 0) \
				+ (1.5 - 0.5*(1-dedt_ce/self.dedt_ce_max)/(1+7.56 * dedt_ce/ self.dedt_ce_max / self.kappa))*(dedt_ce >0)\
				+ (dedt_ce == 0)
	
	def F_va_inverse(self, F):
		return (self.dedt_ce_max * (self.kappa*(F-1)/(F+self.kappa)) * (F <= 1)\
				+ (2 * (-1 + F)) / (3 * 7.56 / self.kappa + 1 - 2 * 7.56 * F / self.kappa) * (F > 1))
	
	def F_lp(self, e_m): 
		'''
		Passive force-length relationship
		'''
		return (2.64 * e_m**2 - 5.3 * e_m + 2.66) * (e_m > 1)

	
	def k_see_fun(self, F, F_0 = 0, l_0 = 0):
		'''
		Function for the stiffness of SEE (mainly effect from tendon)
		'''
		# Constant stiffness
		stiffness = self.k_see * np.ones_like(F)

		# Exponential stiffness Lichtwark and Wilson 2008
		# Constants from Lichtwark and Wilson 2008
		# Q = 20 # Unitless
		# k_l = 325e3 # N/m
		# stiffness = k_l/F_0*l_0 * ( 1 + (0.9/-np.exp(Q * F)))

		return stiffness
	##########################################################
	# 			Function to compute the muscle force 
	def computeForce(self, act_, e_m_, dedt_m_): 
		'''
		Function to compute the scaled muscle force give act, e_m, dedt_m
		'''
		return act_ * self.F_la(e_m_) * self.F_va(dedt_m_) + self.F_lp(e_m_)
	
	##########################################################
	# 			FUNCTIONS TO SCALE FORCE-VELOCITY CONSTANTS
	#			BASED ON ACTIVE MUS
	def setActivationScalings(self, N_act_mat_scaled): 
		'''
		Function to set the scaling parameters for scaling the force-velocity constants
		The actual scaling is not done here, just setting up the parameters
		'''
		# Define the N_act_mat_scaled
		self.N_act_mat_scaled = N_act_mat_scaled

		# Evaluate the force-velocity parameter distribution
		#	this will get the parameters for each MU
		from lib.MUScalingFunctions import muCSADistr, varMUDist

		# Scaled for muscle composition of slow and fast fibres
		dedt_ce_max_slow = self.params['Mechanics_model']['dedt_ce_max_slow']
		dedt_ce_max_fast = self.params['Mechanics_model']['dedt_ce_max_fast']
		kappa_slow = self.params['Mechanics_model']['kappa_slow']
		kappa_fast = self.params['Mechanics_model']['kappa_fast']

		# Define a the array of MU numbers 
		mu_list = np.arange(self.params['N_MU_sim'])

		# Determine the scaling 
		# We want to find the MU number where there would be a change from slow to fast type fibres 
		#   this is assuming there is a hard switch between MU types (unlikely in reality)
		#   In a later step, we can account for a smoother transition
		self.muCSADistr_list = muCSADistr(mu_list, self.params)
		A_mu_dist_cumsum = np.cumsum(self.muCSADistr_list)
		slow_fibre_pcsa = self.params['muscle_pcsa'] * self.params['N_I'] 
		fibre_trans_point = int(np.argmin(np.abs(A_mu_dist_cumsum - slow_fibre_pcsa)))
		# print(f'Fibre-type transition point {fibre_trans_point}')

		# Calculate the scaled values
		self.dedt_ce_max_scaling = varMUDist(mu_list, fibre_trans_point, dedt_ce_max_slow, dedt_ce_max_fast, self.params)
		self.kappa_scaling = varMUDist(mu_list, fibre_trans_point, kappa_slow, kappa_fast, self.params)

		# List to store the scaled values 
		self.dedt_ce_max_list = []
		self.kappa_list = []
		self.t_feval_list = []


	def scaleFVConstants(self, t): 
		'''
		Take as input the scaled relative activation of the MUs 

		Assumes we have set 
		self.t_vals_act 
		self.N_act_mat_scaled 
		'''

		# Get the current activation levels for the MUs 
		idx_t = np.argmin(np.abs(self.t_vals_act - t))

		# Get the array of activations at the index
		mu_act = self.N_act_mat_scaled[idx_t,:]

		# Assign ones if active and zeros if not active
		mu_act_idx = (mu_act > 0)
		# print(mu_act_idx)

		# Get the weighted average each of the parameters
		if np.sum(mu_act_idx) > 0: # Check if there are an MUs active
			
			# dedt_ce_max_ = np.sum(self.muCSADistr_list[mu_act_idx] * self.dedt_ce_max_scaling[mu_act_idx]) / np.sum(self.muCSADistr_list[mu_act_idx])
			# kappa_ = np.sum(self.muCSADistr_list[mu_act_idx] * self.kappa_scaling[mu_act_idx]) / np.sum(self.muCSADistr_list[mu_act_idx])
			# Take average of nonzero mus
			dedt_ce_max_ = np.sum(mu_act * self.dedt_ce_max_scaling) / np.sum(mu_act)
			kappa_ = np.sum(mu_act * self.kappa_scaling) / np.sum(mu_act)

		else: # If there are no MUs active then keep the default value (arbitrary)
			dedt_ce_max_ = self.params['Mechanics_model']['dedt_ce_max_slow']
			kappa_ = self.params['Mechanics_model']['kappa_slow']
		
		# Redefine the force-velocity constants		
		self.dedt_ce_max = dedt_ce_max_
		self.kappa = kappa_

		self.dedt_ce_max_list.append(dedt_ce_max_)
		self.kappa_list.append(kappa_)
		self.t_feval_list.append(t)



	##########################################################
    # 			EXTERNAL FORCING TERMS
	# Define the model 
	# Define the external forcing terms (eg. prescribed mtu stretches)
	def dedt_mtu(self, t_): 
		return np.zeros_like(t_)
	
	##########################################################
	# 			MUSCLE ACTIVATION TERMS 
	def setActivation(self, t_vals_act, act): 
		self.t_vals_act = t_vals_act
		self.act = act + 1e-3
		# if np.any(act) == 0: 
			# self.act = act + 1e-3
		# else: 
			# self.act = act
		
	def getActivation(self, t): 
		return np.interp(t, self.t_vals_act, self.act)
	

	##########################################################
    # 	TERMS FOR NUMERICAL COMPUTATION OF THE MECHANICS
	# Compute the tendon stretch rate 
	def dedt_see(self, t_, e_m_, F_): 
		return (self.l_mtu * self.dedt_mtu(t_) - self.l_m * self.dedt_m_rhs(t_, e_m_, F_)) / self.l_t

	# Function to solve for the force-velo rel in hill model 
	def getFVcomponent(self, t_, e_f_, F_): 
		# NOTE: Need to set the condition based on the activation (prescribed data)
		# return (F_ - self..F_pa(e_f_)) / act(t_) / mech_model.F_la(e_f_) # TODO: Implement the passive component
		act_ = self.getActivation(t_)
		# THIS WORKS
		# FV_ = F_/act_/mech_model.F_la(e_f_) * (F_ > 1e-10) # THIS WORKS (sort of)
		FV_ = F_/act_/self.F_la(e_f_) # NOTE: We assume act_ >0 for all t (do this by applying a small offset)

		return FV_ # Current implementation assumes that there are no lengthening parts of the contraction (e_m <= 1)


	# ODE RHS 
	def dedt_m_rhs(self, t_, e_m_, F_): 
		# Ignore conversion from e_m to e_f
		x_ = self.getFVcomponent(t_, e_m_, F_)                   # Get the force-velocity component of the force
		dedt_m_ = self.F_va_inverse(x_) # Take the inverse to get dedt_m
		return dedt_m_

	def dFdt_rhs(self, t_, e_m_, F_): 
		return  self.k_see_fun(F_) * self.dedt_see(t_, e_m_, F_)

	def rhs(self, t, x): 
		'''
		Right hand side of the ODE
		'''
		# print(f't = {t}')
		assert  not np.isnan(t) , "Invalid time value" # Ensure t is not nan (usually division by 0)

		e_m = x[0]
		F = x[1]
		return (self.dedt_m_rhs(t, e_m, F), self.dFdt_rhs(t, e_m, F))

	def solver(self, tend):
		init_cond = (1, self.getActivation(0)) # Assume the muscle is at rest with zero force 

		# Tspan 
		tspan = (0,tend)

		# Solve the model using solve_ivp 
		from scipy.integrate import solve_ivp
		sol = solve_ivp(self.rhs, tspan, init_cond, 'RK45' , max_step = self.ode_max_step)

		# Unpack solution 
		self.t_sol = sol.t
		self.e_m_sol  = sol.y[0,]
		self.F_m_sol  = sol.y[1,]

		return self.t_sol, self.e_m_sol, self.F_m_sol
	
	def rhsMUPool(self, t, x): 
		'''
		Right hand side of the ODE
		'''
		# print(f't = {t}')
		assert  not np.isnan(t) , "Invalid time value" # Ensure t is not nan (usually division by 0)

		e_m = x[0]
		F = x[1]

		# Revaluate the force-velocity constants the given time 
		self.scaleFVConstants(t)

		return (self.dedt_m_rhs(t, e_m, F), self.dFdt_rhs(t, e_m, F))

	def solverMUPool(self, tend):
		init_cond = (1, self.getActivation(0)) # Assume the muscle is at rest with zero force 

		# Tspan 
		tspan = (0,tend)

		# Solve the model using solve_ivp 
		from scipy.integrate import solve_ivp
		sol = solve_ivp(self.rhsMUPool, tspan, init_cond, 'RK45' , max_step = self.ode_max_step)

		# Unpack solution 
		self.t_sol = sol.t
		self.e_m_sol  = sol.y[0,]
		self.F_m_sol  = sol.y[1,]

		return self.t_sol, self.e_m_sol, self.F_m_sol
	

	######################################################
	#		PLOT THE OUTPUT

	def plotFVParamScaling(self, run_id): 
		# Recalculate the force-velo constants at the time points for plotting
		# dedt_ce_list = []
		# kappa_list = []
		# for t_ in self.t_sol: 
		# 	self.scaleFVConstants(t_)
		# 	dedt_ce_list.append(self.dedt_ce_max)
		# 	kappa_list.append(self.kappa)

		fig,ax = plt.subplots(layout = 'constrained',figsize = (6,4))
		ax.plot(self.t_feval_list, self.dedt_ce_max_list, 'k', label='dedt_ce_max')	
		ax.set_xlabel('Time (s)')
		ax.set_ylabel('dedt_ce_max (1/s)')

		ax_2 = ax.twinx() 
		ax_2.plot(self.t_vals_act,np.sum(self.N_act_mat_scaled, 1))
		ax_2.set_ylabel('Number MUs Active')

		plt.savefig('Figures/' + run_id + '/MechModel_dedt_ce_scaling.pdf')
		plt.close()

		fig,ax = plt.subplots(layout = 'constrained',figsize = (6,4))
		ax.plot(self.t_feval_list, self.kappa_list, 'k', label='kappa')	
		ax.set_xlabel('Time (s)')
		ax.set_ylabel('Kappa (unitless)')
		
		ax_2 = ax.twinx() 
		ax_2.plot(self.t_vals_act,np.sum(self.N_act_mat_scaled, 1))
		ax_2.set_ylabel('Number MUs Active')
		
		plt.savefig('Figures/' + run_id + '/MechModel_kappa.pdf')
		plt.close()

	
	def plotMechanics(self): 		
		fig, ax = plt.subplots(1,4,figsize = (10, 4), layout = 'constrained')

		# Activation and force plot 
		ax_0 = ax[0] 
		ax_0.plot(self.t_sol, self.getActivation(self.t_sol), color = 'k', ls = 'solid', label = 'Activation')
		ax_0.plot(self.t_sol, self.F_m_sol, color = 'k', ls = 'dashed', label = 'Force')
		ax_0.set_xlabel('Time (s)')
		ax_0.set_ylabel('Force/Act (norm)')
		ax_0.legend()

		# Muscle, fibre, and tendon stretches 
		ax_1 = ax[1] 
		ax_1.plot(self.t_sol, self.e_m_sol, color = 'k', ls = 'solid', label = 'Muscle stretch')
		# ax_1.plot(t_sol, convertemtoef(e_m), color = 'k', ls = 'dashed', label = 'Fibre stretch')
		ax_1.set_xlabel('Time (s)')
		ax_1.set_ylabel('Stretch')
		ax_1.legend()

		# Stretch rates
		ax_2 = ax[2] 
		ax_2.plot(self.t_sol, self.dedt_m_rhs(self.t_sol, self.e_m_sol, self.F_m_sol), color = 'k', ls = 'solid', label = 'Muscle stretch rate')
		ax_2.plot(self.t_sol, self.dedt_see(self.t_sol, self.e_m_sol, self.F_m_sol), color = 'k', ls = 'dotted', label = 'Tendon stretch rate')
		ax_2.set_xlabel('Time (s)')
		ax_2.set_ylabel('Stretch')
		ax_2.legend()

		# Plot the force-velocity relationship
		from matplotlib.collections import LineCollection
		x = self.F_m_sol
		x_ = self.getFVcomponent(self.t_sol, self.e_m_sol, self.F_m_sol)
		y = self.F_va_inverse(x_)

		# Change these
		xvals = x
		yvals = y



		points = np.array([xvals, yvals]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)

		ax_3 = ax[3]

		# Create a continuous norm to map from data points to colors
		norm = plt.Normalize(self.t_sol.min(), self.t_sol.max())
		lc = LineCollection(segments, cmap='viridis', norm=norm)
		# lc = LineCollection(segments, cmap='viridis')
		# Set the values used for colormapping
		lc.set_array(self.t_sol)
		lc.set_linewidth(3)
		line = ax_3.add_collection(lc)

		ax_3.set_xlim((min(xvals),max(xvals)))
		ax_3.set_ylim((min(yvals),max(yvals)))
		ax_3.set_xlabel('Force (N)')
		ax_3.set_ylabel('Stretch rate')

		# plt.show()
		plt.savefig('Figures/' + self.params['run_id'] + '/MechResults.jpg')
		plt.close()


