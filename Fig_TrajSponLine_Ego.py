###
# IMPORT
###
import BrainMove as bm

###
# SIMULATION
###
# Parameters of simulation
deterministic = True
fig_name = "TrajSponLine_Ego"
if deterministic:
    env_prop = bm.EnvProp(time_max=400,time_memory=300)
    beh_prop = [bm.BehProp(allo=False,omega0=0.1,init_hold=100)]
    seed = 0
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=1e-3,time_max=400,time_memory=300)
    beh_prop = [bm.BehProp(N=10,angle_num=16,allo=False,omega0=0.1,init_hold=100)]
    seed = 15
ind_prop = [bm.IndProp(beh_prop)]
# Run the simulation
bm.snapshot_trajectory(env_prop,ind_prop,['Reds'],fig_name,True,seed,deterministic,allocentric=False)

