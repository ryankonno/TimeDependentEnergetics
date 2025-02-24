# Time-dependent energetics 

This repository contains code to analyze individual motor unit contributions to the energetic cost of muscle contraction by accounting for both time-dependent initial an recovery processes.


## Codes 


### runFullEnergetics.py 
Runs the full (initial and recovery) models to compute the energetic cost given experimentally measured motor unit firing times. 

### runBioenergetics<*>.py
Runs only the recovery model 
#### <*> = Phillips1993
Reproduce experiment from Phillips et al. 1993. 
#### <*> = Phillips1993OptParams
Optimizes parameters to the experimental data 
#### <*> = Barclay 1995 
Reproduces the experiment from Barclay et al. 1995. This code is suplemented with computeParametersBarclay1995.py which computes the necessary parameters for the simulation.

### runEnergetics_EXP2.py 
Runs a simulation to compare the influence of firing rate and MU size between two motor units, where the firing rate of the smaller motor unit is optimized to match the force of the larger MU. 


