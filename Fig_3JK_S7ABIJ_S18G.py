###
# IMPORT
###
import BrainMove as bm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

###
# SIMULATION
###
# Parameter to determine the figure panel: Fig. 3J,K = False, Fig. 13G = False, Fig. S7A,B = False, Fig. S7I,J = True
deterministic = True

# Parameters of simulation
fig_name = "TrajSponInh_Allo"
if deterministic:
    env_prop = bm.EnvProp(time_max=500,time_memory=500)
    beh_prop = [bm.BehProp(omega0=30,init_hold=0,S=np.exp(-5))]
    seed = 0
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=0.005,time_max=500,time_memory=500)
    beh_prop = [bm.BehProp(N=10,angle_num=16,omega0=30,init_hold=0,S=np.exp(-5))]
    seed = 2
ind_prop = [bm.IndProp(beh_prop)]
# Create a colormap
yellows = LinearSegmentedColormap.from_list("Yellows", [(0.0,(1,1,0.8)), (0.2,(0.8, 0.6, 0)),(1.0,(0.4, 0.3, 0.00))])
# Run the simulation
bm.snapshot_trajectory(env_prop,ind_prop,[yellows],fig_name,True,seed,deterministic,ticks=False,stat_t_lag=250)

