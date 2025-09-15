###
# IMPORT
###
import BrainMove as bm

###
# SIMULATION
###
# Parameter to determine the figure panel: Fig. 3A,B = False, Fig. 13A = False, Fig. S6A,B = True
deterministic = False

# Parameters of simulation
fig_name = "TrajSponLine_Allo"
if deterministic:
    env_prop = bm.EnvProp(time_max=1000, time_memory=1000)
    beh_prop = [bm.BehProp(omega0=0.1, init_hold=0)]
    seed = 0
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=0.005,time_max=1000,time_memory=1000)
    beh_prop = [bm.BehProp(N=10,angle_num=16,omega0=0.1,init_hold=0)]
    seed = 15
ind_prop = [bm.IndProp(beh_prop)]
# Run the simulation
bm.snapshot_trajectory(env_prop,ind_prop,['Reds'],fig_name,True,seed,deterministic,ticks=False,stat_t_lag=500)

