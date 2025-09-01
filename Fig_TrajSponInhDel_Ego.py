###
# IMPORT
###
import BrainMove as bm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

###
# SIMULATION
###
# Parameters of simulation
deterministic = True
fig_name = "TrajSponInhDel_Ego"
if deterministic:
    env_prop = bm.EnvProp(time_max=400, time_memory=300)
    beh_prop = [bm.BehProp(omega0=30, init_hold=100, S=np.exp(-5), allo=False, tau=20)]
    seed = 0
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=1e-3,time_max=400,time_memory=300)
    beh_prop = [bm.BehProp(N=10,angle_num=16,allo=False,omega0=30,init_hold=100,S=np.exp(-5),tau=20)]
    seed = 2
ind_prop = [bm.IndProp(beh_prop)]
# Create a colormap
yellows = LinearSegmentedColormap.from_list("Yellows", [(1,1,0.8), (0.8, 0.6, 0)])
# Run the simulation
bm.snapshot_trajectory(env_prop,ind_prop,[yellows],fig_name,True,seed,deterministic,allocentric=False)

