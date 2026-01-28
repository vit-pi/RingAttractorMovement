###
# IMPORT
###
import BrainMove as bm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


###
# SIMULATION
###
# Parameters of simulation
deterministic = False
fig_name = "TrajPeriodicLevi_Allo"
env_prop = bm.EnvProp(angle_num=16,time_step=1e-3)
beh_prop = [bm.BehProp(N=10,angle_num=16,omega0=100,S_update_rule=[1,0.1])]
ind_prop = [bm.IndProp(beh_prop)]
# Create a colormap
yellows = LinearSegmentedColormap.from_list("Yellows", [(0.0,(1,1,0.8)), (0.2,(0.8, 0.6, 0)),(1.0,(0.4, 0.3, 0.00))])
# Run the simulation
bm.snapshot_trajectory(env_prop,ind_prop,['Reds',yellows],fig_name,True,0,deterministic,plot_S=True,stat_t_lag=150)

