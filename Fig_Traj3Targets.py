###
# IMPORT
###
import BrainMove as bm

###
# SIMULATION
###
# Parameters of simulation
deterministic = False
fig_name = "Traj3Target"
cue = [[0,False,10,1,15,12],[0,False,10,1,15,-12],[0,False,10,1,25,0]]
if deterministic:
    env_prop = bm.EnvProp()
    beh_prop = [bm.BehProp(cue=cue)]
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=1e-3)
    beh_prop = [bm.BehProp(cue=cue,N=10,angle_num=16)]
ind_prop = [bm.IndProp(beh_prop)]
# Run the simulation
bm.snapshot_trajectory(env_prop,ind_prop,['Blues'],fig_name,True,0,deterministic)

