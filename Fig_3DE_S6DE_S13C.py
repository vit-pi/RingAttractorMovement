###
# IMPORT
###
import BrainMove as bm

###
# SIMULATION
###
# Parameter to determine the figure panel: Fig. 3D,E = False, Fig. 13C = False, Fig. S6D,E = True
deterministic = False

# Parameters of simulation
fig_name = "TrajSponWavy_Allo"
if deterministic:
    env_prop = bm.EnvProp(time_max=500,time_memory=500)
    beh_prop = [bm.BehProp(omega0=1,tau=20,init_hold=0)]
    seed = 2
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=0.005,time_max=500,time_memory=200)
    beh_prop = [bm.BehProp(N=10,angle_num=16,omega0=1,tau=20,init_hold=0)]
    seed = 4
ind_prop = [bm.IndProp(beh_prop)]
# Run the simulation
bm.snapshot_trajectory(env_prop,ind_prop,['Reds'],fig_name,True,seed,deterministic,ticks=False,stat_t_lag=150)

