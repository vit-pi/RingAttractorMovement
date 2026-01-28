###
# IMPORT
###
import BrainMove as bm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

###
# SIMULATION
###
# Parameter to determine the figure panel: Fig. 3M,O = False, Fig. 13J = False, Fig. S8D,F = False, Fig. S7L,N = True
deterministic = True

# Parameters of simulation
fig_name = "TrajSponDownBump_Ego"
if deterministic:
    env_prop = bm.EnvProp(time_max=5, time_memory=5)
    beh_prop = [bm.BehProp(allo=False, omega0=10,v0=100,init_hold=0,S=np.exp(-5))]
    beh_prop[0].init_brain = [3, beh_prop[0].brain_fp, 1, np.pi / 2, np.pi / 2]
    seed = 0
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=0.005,time_max=5 ,time_memory=5)
    beh_prop = [bm.BehProp(N=10,angle_num=16,allo=False,omega0=10,v0=100,init_hold=0,S=np.exp(-5))]
    beh_prop[0].init_brain = [3, beh_prop[0].brain_fp, 1, np.pi / 2, np.pi/2]
    seed = 2
ind_prop = [bm.IndProp(beh_prop)]
# Create a colormap
yellows = LinearSegmentedColormap.from_list("Yellows", [(0.0,(1,1,0.8)), (0.2,(0.8, 0.6, 0)),(1.0,(0.4, 0.3, 0.00))])
# Run the simulation
bm.snapshot_trajectory(env_prop,ind_prop,[yellows],fig_name,True,seed,deterministic,allocentric=False,ticks=False,stat_t_lag=3)

