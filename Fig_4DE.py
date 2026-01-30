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
fig_name = "TrajTrans_SearchCue_Allo"
cue = [[3,False,2,1,0,0]]
env_prop = bm.EnvProp(angle_num=16,time_step=0.005,time_max=1000,time_memory=1000)
beh_prop = [bm.BehProp(N=10,angle_num=16,v0=0.5,omega0=10,init_hold=0,S=np.exp(-5),S_update_rule=[7,np.exp(-3),np.exp(-5),0.17,50],cue=cue)]
ind_prop = [bm.IndProp(beh_prop)]
# Create a colormap
yellows = LinearSegmentedColormap.from_list("Yellows", [(0.0,(1,1,0.8)), (0.2,(0.8, 0.6, 0)),(1.0,(0.4, 0.3, 0.00))])
# Run the simulation
bm.snapshot_trajectory(env_prop,ind_prop,["Blues",yellows],fig_name,True,3,deterministic,plot_S=True,stat_t_lag=150,gradient=True,S_lim=(0,0.2))

