###
# IMPORT
###
import BrainMove as bm

###
# SIMULATION
###
# Parameters of simulation
deterministic = False
fig_name = "Traj1Target_Trace_SpCue"
cue = [[3,False,2,1,35,0]]
if deterministic:
    env_prop = bm.EnvProp()
    beh_prop = [bm.BehProp(cue=cue)]
    repeats = 50
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=1e-3)
    beh_prop = [bm.BehProp(cue=cue,N=10,angle_num=16)]
    repeats = 1
ind_prop = [bm.IndProp(beh_prop)]
# Run the simulation
bm.snapshot_targets(env_prop,ind_prop,['Blues'],fig_name,True,0,deterministic,repeats=repeats)

