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
fig_name = "TrajSponSpiral_Ego"
env_prop = bm.EnvProp(angle_num=16,time_step=1e-3)
beh_prop = [bm.BehProp(N=10,angle_num=16,allo=False,omega0=0.1),bm.BehProp(N=16,angle_num=16,S=np.exp(-5),t_neuro=50,allo=False,omega0=0.1)]
trans = np.asarray([[0, 0.1], [0, 0]])
ind_prop = [bm.IndProp(beh_prop,trans)]
# Create a colormap
yellows = LinearSegmentedColormap.from_list("Yellows", [(1,1,0.8), (0.8, 0.6, 0)])
# Run the simulation
bm.snapshot_trajectory(env_prop,ind_prop,['Reds',yellows],fig_name,True,1,deterministic,allocentric=False)

