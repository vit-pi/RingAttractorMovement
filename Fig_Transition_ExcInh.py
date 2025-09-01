###
# IMPORT
###
import BrainMove as bm

###
# SIMULATION
###
# Parameters of simulation
fig_name = "ExcInhTransition_Final"
env_prop = bm.EnvProp(time_max=20,angle_num=64)
# Run the simulation
bm.plot_exc_inh_phase_heatmap(env_prop,fig_name,0,None)