###
# IMPORT
###
import BrainMove as bm

###
# SIMULATION
###
# Parameter to determine the figure panel: Fig. 2A-B = True, Fig. S2A-B = False
deterministic = False

# Parameters of simulation
fig_name = "TrajBlinkTarget"
cue = [[4,False,1,1,18,10,10,10,0,0]]
if deterministic:
    env_prop = bm.EnvProp()
    beh_prop = [bm.BehProp(cue=cue)]
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=0.005)
    beh_prop = [bm.BehProp(cue=cue,N=10,angle_num=16)]
ind_prop = [bm.IndProp(beh_prop)]
# Run the simulation
bm.snapshot_trajectory(env_prop,ind_prop,['Blues','Reds'],fig_name,True,0,deterministic)

