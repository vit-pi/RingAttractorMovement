###
# IMPORT
###
import BrainMove as bm

###
# SIMULATION
###
# Parameters of simulation
fig_name = "Transition_Phases"
env_prop = bm.EnvProp(time_max=20,angle_num=64)
kappa_max = 0.5
# Run the simulation
bm.plot_phases_heatmap(env_prop,kappa_max,fig_name,0,None)