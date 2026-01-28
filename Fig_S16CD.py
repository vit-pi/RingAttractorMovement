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
cue = [[3,False,2,1,30,15],[3,False,2,1,30,-15]]
deterministic = False
fig_name = "TrajTargetSwitch_Allo_Trial"
env_prop = bm.EnvProp(angle_num=16,time_step=0.005)
beh_prop = [bm.BehProp(N=10,angle_num=16,cue=cue),bm.BehProp(N=10,angle_num=16)]
trans = np.asarray([[0, 0.1], [0.1, 0]])
ind_prop = [bm.IndProp(beh_prop,trans)]
# Run the simulation
bm.snapshot_targets(env_prop,ind_prop,['Blues','Reds'],fig_name,True,1,deterministic,repeats=10)

