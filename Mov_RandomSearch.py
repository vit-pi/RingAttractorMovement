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
fig_name = "Mov_RandomSearch"
if deterministic:
    env_prop = bm.EnvProp(time_max=500,time_memory=500)
    beh_prop = [bm.BehProp(omega0=30,init_hold=50,S=np.exp(-5),init_pos=(50,50))]
    seed = 0
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=0.005,time_max=500,time_memory=500,pos_max=(10,10))
    beh_prop = [bm.BehProp(N=10,angle_num=16,omega0=30,init_hold=0,S=np.exp(-5),init_pos=(5.5,6.5))]
    seed = 2
ind_prop = [bm.IndProp(beh_prop)]
# Create a colormap
yellows = LinearSegmentedColormap.from_list("Yellows", [(0.0,(1,1,0.8)), (0.2,(0.8, 0.6, 0)),(1.0,(0.4, 0.3, 0.00))])
# Run the simulation
bm.movie_snapshots(env_prop,ind_prop,[yellows],2,True,True,fig_name,15, deterministic)