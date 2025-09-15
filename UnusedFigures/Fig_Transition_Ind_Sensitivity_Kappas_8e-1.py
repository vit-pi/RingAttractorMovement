###
# IMPORT
###
import BrainMove as bm
import numpy as np

###
# SIMULATION
###
# Parameters of simulation
fig_name = "Transition_Ind_Sensitivity_Kappas8e-1"
env_prop = bm.EnvProp(time_max=20)
beh_prop = bm.BehProp()
s=[0.8,1]
kappa_max=10
# Run the simulation
bm.plot_phases_ind_kappas(env_prop,beh_prop,np.pi,s,kappa_max,fig_name,0)