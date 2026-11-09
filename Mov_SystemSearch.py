###
# IMPORT
###
import BrainMove as bm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

###
# SIMULATION
###
# Parameter to determine the figure panel: Fig. 3G,I = False, Fig. 13F = False, Fig. S6G,I = True
deterministic = False

# Parameters of simulation
fig_name = "Mov_SystemSearch"
if deterministic:
    env_prop = bm.EnvProp(time_max=500,time_memory=500)
    beh_prop = [bm.BehProp(omega0=1,tau=20,init_hold=0)]
    seed = 2
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=0.005,time_max=500,time_memory=500,pos_max=(40,40))
    beh_prop = [bm.BehProp(N=10,angle_num=16,omega0=1,tau=20,init_hold=0,init_pos=(3,3))]
    seed = 4
ind_prop = [bm.IndProp(beh_prop)]
# Run the simulation
bm.movie_snapshots(env_prop,ind_prop,["Reds"],2,True,True,fig_name,0, deterministic)