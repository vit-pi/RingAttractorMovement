###
# IMPORT
###
import BrainMove as bm

###
# SIMULATION
###
# Parameter to determine the figure panel: Fig. 2C-D = True, Fig. S2C-D = False
deterministic = False

# Parameters of simulation
fig_name = "Mov_Decision"
cue = [[3,False,2,1,35,35],[3,False,2,1,35,5]]
if deterministic:
    env_prop = bm.EnvProp()
    beh_prop = [bm.BehProp(cue=cue)]
    repeats = 50
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=0.005,pos_max=(40,40))
    beh_prop = [bm.BehProp(cue=cue,N=10,angle_num=16,init_pos=(5,20))]
    repeats = 50
ind_prop = [bm.IndProp(beh_prop)]
# Run the simulation
bm.movie_snapshots(env_prop,ind_prop,['Blues'],2,True,True,fig_name,0,deterministic)