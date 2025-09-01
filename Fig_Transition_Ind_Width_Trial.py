###
# IMPORT
###
import BrainMove as bm
import numpy as np
import matplotlib.pyplot as plt

###
# SIMULATION
###
# Parameters of simulation
fig_name = "IndTransition_Width_Trial"
env_prop = bm.EnvProp(time_max=20)
beh_prop = bm.BehProp()
# Run the simulation
bm.plot_ind_phase_heatmap(env_prop, beh_prop,  [0.5,1],[2.85480092093160,5],fig_name,0)