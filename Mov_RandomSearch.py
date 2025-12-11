###
# IMPORT
###
import BrainMove as bm

###
# SIMULATION
###
# Parameter to determine the figure panel: Fig. 3G,I = False, Fig. 13F = False, Fig. S6G,I = True
deterministic = False

# Parameters of simulation
fig_name = "Mov_RandomSearch"
if deterministic:
    env_prop = bm.EnvProp(time_max=1000, time_memory=1000)
    beh_prop = [bm.BehProp(omega0=1, T=3, init_hold=0,allo=False)]
    seed = 1
else:
    env_prop = bm.EnvProp(angle_num=16,time_step=0.005,time_max=1000,time_memory=1000)
    beh_prop = [bm.BehProp(N=10,angle_num=16,omega0=1,T=3,init_hold=0,allo=False)]
    seed = 15
ind_prop = [bm.IndProp(beh_prop)]
# Run the simulation
bm.movie_snapshots(env_prop,ind_prop,['Reds'],10,True,True,fig_name,15)