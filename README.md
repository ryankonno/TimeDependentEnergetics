# Time-dependent energetics 

This repository contains the time-dependent energetics model developed in the manuscript 
    
 - Konno RN, Lichtwark GA, Dick TJM. A time-dependent mechano-bioenergetics model of muscle contraction. 2026. Preprint available at [PREPRINT LINK HERE]

This model is designed to capture the time-dependent response of muscle energy consumption under a number of contractile conditions. As a validation we include codes demonstrating model behaviour in response to dynamic contractions, varying stimuli frequencies, and twitch contractions. 

All codes required to run the model are contained within this repository. 

If you have any questions of find any issues in this repository please email r.konno@uq.edu.au.

## Model 
The model framework consists of four components including an excitation-activation model, a mechanics model, an initial energetics model, and a recovery bioenergetics model. The models are contained in the following codes, respectively, `ActivationModel.py`, `MechanicsModel.py`, `InitialEnergeticsModel.py`, and `BioenergeticsModel.py`. For a description of the model, see [PREPRINT LINK HERE]

## Codes 
The codes contained within this repository include demo codes along with parameter optimisation and validation codes. The relevant codes are described below. 

### DEMO codes 
#### ```DEMO_workrest_bioenergetics.py```
This script simulates the energetics of a 'work-rest' contraction. Despite the name this is an isometric contraction. 

The 'work' phase consists of a fixed initial energetic rate prescribed (E_rate). Following the initial period, the recovery of the energetic rates is computed using the bioenergetics model.

All parameters used by the model are defined in this file. The params dictionary contains the time settings used to configure the simulation together with muscle-specific energetics parameters for SOL and EDL.

The script produces three figures and prints a summary of the fitted recovery behaviour to the console:

 - Figure 1: Cumulative total energy over the simulation

 - Figure 2: Recovery energy rate after the work phase with the fitted exponential decay

 - Figure 3: ATP and PCr concentration over time

 - Table 1: The fitted time constant, initial energy, recovery energy, and recovery-to-initial energy ratio for each target energy value.

#### ```DEMO_repeatedcontractions.py```
This code simulates the energetics for repeated isometric contractions. 

All parameters for the code are contained within the file. 

Within the parameter dictionary (params) are time variables to set up the simulation including the contraction frequency and stimulation frequency. Muscle specific parameter for the energetics model are also included for the SOL and EDL. 

To investigate the role of contraction frequency or stimulation frequency, the parameter sim_type can be chosen to choose the simulation setup. The options are 
 - varycontrfreq: vary the frequency of the contractions with stimulation frequency as defined
 - varystimfreq: vary the frequency of the stimulation with contraction frequency as defined 
 - single: Perfrom a single stimulation and contraction frequency with parameters as defined


The outputs from this code include two tables and two figures: 

 - Table 1: Comparison of time constants and recovery to initial energetics across different frequencies 

 - Table 2: Total energy be component over the simulation 

 - Figure 1: Cumulative energy over the simulation including total energy (solid line), initial energy (dotted line), and recovery energy (dashed line)

 - Figure 2: Total energy spent per componentent

NOTE: for output, the frequency will correspond to either contraction frequency or stimulation frequency depending on the sim_type parameter

### Parameter optimisation 

#### ```1_initial_energy_opt.py```
Initial energetics model optimisation was done to optimise the initial energetics parameters to data from 
 - Barclay, C. J., Woledge, R. C. and Curtin, N. A. (2010). Is the efficiency of mammalian (mouse) skeletal muscle temperature dependent? J Physiol. 588, 3819–3831.


#### ```2_recovery_energy_opt.py```
Bioenergetics model optimisation was used to inform parameters based on the time course of recovery observed in 
 - Barclay, C. J., Arnold, P. D. and Gibbs, C. L. (1995). Fatigue and heat production in repeated contractions of mouse skeletal muscle. J Physiol. 488, 741–752.

The code `2_recovery_energy_run.py` will run the script with optimised parameters, so that the whole optimisation does not need to be rerun. 


Note parameters were corrected post optimisation to account for temperature scaling and likely overapproximation of recovery heat in Barclay et al., 1995. 

### Validation
The model was validated against data from a number of experimental studies: 

 - Barclay, C. J. and Weber, C. L. (2004). Slow skeletal muscles of the mouse have greater initial efficiency than fast muscles but the same net efficiency. J. Physiol. 559, 519–533.

 - Barclay, C. J. (2012). Quantifying Ca2+ release and inactivation of Ca2+ release in fast- and slow-twitch muscles. J Physiol. 590, 6199.

 - Lewis, D. B. and Barclay, C. J. (2014). Efficiency and cross-bridge work output of skeletal muscle is decreased at low levels of activation. Pflugers Arch. 466, 599–609.

 - Mast, F. and Elzinga, G. (1987). Time course of aerobic recovery after contraction of rabbit papillary muscle. Am J Physiol 253, H325-332.

 - Mast, F. and Elzinga, G. (1988). Recovery heat production of isolated rabbit papillary muscle at 20°C. Pflugers Arch. 411, 600–605.

All muscle specific parameters are contained within `parameters_valid.py`

#### ```3_validation_B2012```
This code implements a comparison to the dataset from Barclay et al., 2012. These simulations include a series of two muscle twitches at varying frequency to test the dependence of activation heat rates (heat due to Ca transport) on the the frequency of muscle twitches. Simulations are performed for both fast-type and slow-type muscle.

#### ```3_validation_BW2004```
This code implements a comparison to the dataset from Barclay and Weber 2004. These simulations include dynamic contractions with an isometric phase followed by an isokinetic phase. Simulations are performed at varying contraction frequencies. Simulations are performed for both fast-type and slow-type muscle.

#### ```3_validation_LB2014```
This code implements a comparison to the dataset from Lewis and Barclay 2004. These simulations follow the same protocol as BW2004 simulations, but only at one contraction frequency. Here, the contraction frequency is varied to investigate the role of submaximal activations on energetic rates. Simulations are performed for both fast-type and slow-type muscle.

#### ```3_validation_ME1987```
This code implements a comparison to the dataset from Mast and Elzinga 1987. The protocol for this simulations involves a series of isometric muscle twitches. The time-constant of recovery is determined to compare with the time-constant of oxygen recovery. Slow-type muscle parameters are used.

#### ```3_validation_ME1988```
This code implements a comparison to the dataset from Mast and Elzinga 1988. The protocol for this simulations involves a series of 10 isometric muscle twitches. The time-constant of recovery and the ratio of initial to recovery heat is compared to the experimental data. Slow-type muscle parameters are used.

## AI use acknowledgement
Portions of the codes within this repository were written with the assistance of Microsoft Copilot. All codes have been manually checked and tested.