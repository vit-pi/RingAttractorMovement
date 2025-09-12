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
fig_name = "TrajSponUpBump_Allo"
if deterministic:
    env_prop = bm.EnvProp(time_max=5, time_memory=5)
    beh_prop = [bm.BehProp(omega0=10,v0=100,init_hold=0, S=np.exp(-5))]
    beh_prop[0].init_brain = [3, beh_prop[0].brain_fp, 1, np.pi / 2, np.pi / 2]
    seed = 0
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=0.005,time_max=5, time_memory=5)
    beh_prop = [bm.BehProp(N=10,angle_num=16,omega0=10,v0=100,init_hold=0)]
    beh_prop[0].init_brain = [3, beh_prop[0].brain_fp, 1, np.pi / 2, np.pi/2]
    seed = 2
ind_prop = [bm.IndProp(beh_prop)]
# Run the simulation
bm.snapshot_trajectory(env_prop,ind_prop,['Reds'],fig_name,True,seed,deterministic,ticks=False, stat_t_lag=3)

