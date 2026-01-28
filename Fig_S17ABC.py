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
cue=[[3,False,2,1,80,0]]
fig_name = "TrajSwitch_MoreAllo_AlloEgo"
env_prop = bm.EnvProp(angle_num=16,time_step=1e-3,time_max=200,time_memory=200)
beh_prop = [bm.BehProp(N=10,angle_num=16,cue=cue, allo=True, v0=0.5),bm.BehProp(N=10,angle_num=16,S=np.exp(-5),omega0=20,allo=False, v0=0.5)]
trans = np.asarray([[0, 0.01], [0.1, 0]])
ind_prop = [bm.IndProp(beh_prop,trans)]
# Create a colormap
yellows = LinearSegmentedColormap.from_list("Yellows", [(0.0,(1,1,0.8)), (0.2,(0.8, 0.6, 0)),(1.0,(0.4, 0.3, 0.00))])
# Run the simulation
bm.snapshot_trajectory(env_prop,ind_prop,['Blues',yellows],fig_name,True,0,deterministic,stat_t_lag=150)

