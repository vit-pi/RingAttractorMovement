###
# IMPORT
###
import BrainMove as bm

###
# SIMULATION
###
# Parameter to determine the figure panel: Fig. 3D,F = False, Fig. 13D = False, Fig. S6D,F = True
deterministic = False

# Parameters of simulation
fig_name = "TrajSponWavy_Ego"
if deterministic:
    env_prop = bm.EnvProp(time_max=500,time_memory=500)
    beh_prop = [bm.BehProp(allo=False,omega0=1,tau=20,init_hold=0)]
    seed = 2
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=0.005,time_max=500,time_memory=200)
    beh_prop = [bm.BehProp(N=10,angle_num=16,allo=False,omega0=1,tau=20,init_hold=0)]
    seed = 4
ind_prop = [bm.IndProp(beh_prop)]
# Run the simulation
bm.snapshot_trajectory(env_prop,ind_prop,['Reds'],fig_name,True,seed,deterministic,allocentric=False,ticks=False,stat_t_lag=150)

