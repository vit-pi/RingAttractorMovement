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
cue = [[5,False,2,1,30,15,0.1,True,0],[5,False,2,1,30,-15,0.1,True,0]]
deterministic = False
fig_name = "TrajTargetSwitch_Two_Allo"
env_prop = bm.EnvProp(angle_num=16,time_step=0.005)
beh_prop = [bm.BehProp(N=10,angle_num=16,cue=cue)]
ind_prop = [bm.IndProp(beh_prop)]
# Run the simulation
bm.snapshot_targets(env_prop,ind_prop,['Reds','Blues'],fig_name,True,0,deterministic,repeats=10)

