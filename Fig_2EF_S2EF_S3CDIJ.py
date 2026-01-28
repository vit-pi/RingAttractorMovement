###
# IMPORT
###
import BrainMove as bm

###
# SIMULATION
###
# Parameter to determine the figure panel: Fig. 2E-F = True, Fig. S3C-D = True, Fig. S2E-F = False, Fig. S3I-J = False
deterministic = True

# Parameters of simulation
fig_name = "Traj3Target_Trace_SpCue"
cue = [[3,False,2,1,22,15],[3,False,2,1,22,-15],[3,False,2,1,33,0]]
if deterministic:
    env_prop = bm.EnvProp()
    beh_prop = [bm.BehProp(cue=cue)]
    repeats = 50
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=0.005)
    beh_prop = [bm.BehProp(cue=cue,N=10,angle_num=16)]
    repeats = 50
ind_prop = [bm.IndProp(beh_prop)]
# Run the simulation
bm.snapshot_targets(env_prop,ind_prop,['Blues'],fig_name,True,0,deterministic,repeats=repeats)

