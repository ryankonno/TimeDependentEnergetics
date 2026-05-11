# Time-dependent energetics 

This repository contains the time-dependent energetics model developed in the manuscript 
    
 - Konno RN, Lichtwark GA, Dick TJM. A time-dependent mechano-bioenergetics model of muscle contraction. 2026. [Submitted]. Preprint available at [PREPRINT LINK HERE]

This model is designed to capture the time-dependent response of muscle energy consumption under a number of contractile conditions. As a validation we include codes demonstrating model behaviour in response to dynamic contractions, varying stimuli frequencies, and twitch contractions. 

All codes required to run the model are contained within this repository. 

Report any issues in this repository to r.konno@uq.edu.au.

## Model 
The model framework consists of four components including an excitation-activation model, a mechanics model, an initial energetics model, and a recovery bioenergetics model. The models are contained in the following codes, respectively, `ActivationModel.py`, `MechanicsModel.py`, `InitialEnergeticsModel.py`, and `BioenergeticsModel.py`. For a description of the model, see [PREPRINT LINK HERE]

## Codes 
The codes contained within this repository include both parameter optimisation and validation codes. The relevant codes are described below. 

### Parameter optimisation 

##### 1_initial_energy_opt.py
Initial energetics model optimisation was done to optimise the initial energetics parameters to data from 
 - Barclay, C. J., Woledge, R. C. and Curtin, N. A. (2010). Is the efficiency of mammalian (mouse) skeletal muscle temperature dependent? J Physiol. 588, 3819–3831.


##### 2_recovery_energy_opt.py
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

#### 3_validation_B2012
This code implements a comparison to the dataset from Barclay et al., 2012. These simulations include a series of two muscle twitches at varying frequency to test the dependence of activation heat rates (heat due to Ca transport) on the the frequency of muscle twitches. Simulations are performed for both fast-type and slow-type muscle.

#### 3_validation_BW2004
This code implements a comparison to the dataset from Barclay and Weber 2004. These simulations include dynamic contractions with an isometric phase followed by an isokinetic phase. Simulations are performed at varying contraction frequencies. Simulations are performed for both fast-type and slow-type muscle.

#### 3_validation_LB2014
This code implements a comparison to the dataset from Lewis and Barclay 2004. These simulations follow the same protocol as BW2004 simulations, but only at one contraction frequency. Here, the contraction frequency is varied to investigate the role of submaximal activations on energetic rates. Simulations are performed for both fast-type and slow-type muscle.

#### 3_validation_ME1987
This code implements a comparison to the dataset from Mast and Elzinga 1987. The protocol for this simulations involves a series of isometric muscle twitches. The time-constant of recovery is determined to compare with the time-constant of oxygen recovery. Slow-type muscle parameters are used.

#### 3_validation_ME1988
This code implements a comparison to the dataset from Mast and Elzinga 1988. The protocol for this simulations involves a series of 10 isometric muscle twitches. The time-constant of recovery and the ratio of initial to recovery heat is compared to the experimental data. Slow-type muscle parameters are used.

## AI use
The portions of the codes within this repository were written with the assistance of Microsoft Copilot. All codes have been manually checked and tested.