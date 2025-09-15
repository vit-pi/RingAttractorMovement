###
# IMPORT
###
import BrainMove as bm
import numpy as np

###
# SIMULATION
###
# Parameters of simulation
fig_name = "Transition_Ind_Sensitivity_Distance_5e-1"
env_prop = bm.EnvProp(time_max=20)
beh_prop = bm.BehProp()
s=[1,0.5]
# Run the simulation
bm.plot_phases_ind_distance(env_prop,beh_prop,2,s,10,fig_name,0)