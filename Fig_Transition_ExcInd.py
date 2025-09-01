###
# IMPORT
###
import BrainMove as bm

###
# SIMULATION
###
# Parameters of simulation
fig_name = "ExcIndTransition_Final"
env_prop = bm.EnvProp(time_max=20,angle_num=64)
kappa_max = 0.5
# Run the simulation
bm.plot_exc_ind_phase_heatmap(env_prop,kappa_max,fig_name,0,None)