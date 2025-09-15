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
fig_name = "TrajSponInhDel_Ego"
if deterministic:
    env_prop = bm.EnvProp(time_max=500,time_memory=500)
    beh_prop = [bm.BehProp(allo=False,omega0=30,init_hold=0,S=np.exp(-5),tau=20)]
    seed = 0
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=0.005,time_max=500,time_memory=500)
    beh_prop = [bm.BehProp(allo=False,N=10,angle_num=16,omega0=30,init_hold=0,S=np.exp(-5),tau=20)]
    seed = 2
ind_prop = [bm.IndProp(beh_prop)]
# Create a colormap
yellows = LinearSegmentedColormap.from_list("Yellows", [(1,1,0.8), (0.8, 0.6, 0)])
# Run the simulation
bm.snapshot_trajectory(env_prop,ind_prop,[yellows],fig_name,True,seed,deterministic,ticks=False)

