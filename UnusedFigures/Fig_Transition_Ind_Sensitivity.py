###
# IMPORT
###
import BrainMove as bm
import numpy as np

###
# SIMULATION
###
# Parameters of simulation
fig_name = "Transition_Ind_Sensitivity"
env_prop = bm.EnvProp(time_max=20)
beh_prop = bm.BehProp()
s=0.5
kappa=1.5
# Run the simulation
bm.plot_phases_ind_s_kappa(env_prop,beh_prop,np.pi,s,kappa,fig_name,0)