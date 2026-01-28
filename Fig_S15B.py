###
# IMPORT
###
import BrainMove as bm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


###
# SIMULATION
###

# Parameters of simulation
cue = [[3,False,2,1,-15,0],[3,False,2,1,15,0]]
deterministic = True
fig_name = "TrajDecisionMaking"
ic_num = 18
seed = 0

# Prepare random sampling
if seed is not None:
    np.random.seed(seed)

# Create figure
cell_size = 2.5
fig, ax = plt.subplots(figsize=(cell_size,cell_size/1.9), layout="constrained")

# Run the simulation for each trajectory
for i in range(ic_num):
    # prepare parameters
    init_pos = (40*np.cos(np.pi*i/(ic_num-1)),-40*np.sin(np.pi*i/(ic_num-1)))
    env_prop = bm.EnvProp(time_max=600, time_memory=600)
    beh_prop = [bm.BehProp(angle_num=16,cue=cue, init_pos=init_pos)]
    ind_prop = [bm.IndProp(beh_prop)]
    # create a simulation and save its properties
    pattern = bm.Pattern(env_prop, ind_prop)
    pattern.save_properties("Figures//Metadata//",fig_name)
    # do the simulation
    print("Simulating step ", i+1, " of ", ic_num, " ...")
    while pattern.time <= pattern.env_prop.time_max and pattern.get_minimal_cue_distance() > 2:
        pattern.update_pattern(deterministic)
    # do the plots
    for ind in range(pattern.env_prop.ind_num):
        traj_x = np.roll(pattern.nx[:, ind, pattern.env_prop.angle_num], -pattern.pt_now - 1)
        traj_y = np.roll(pattern.nx[:, ind, pattern.env_prop.angle_num + 1], -pattern.pt_now - 1)
        ax.plot(traj_x, traj_y, color="black", linewidth=0.7, zorder=0)
 # stimuli
for c in cue:
    x,y = c[4],c[5]
    radius = c[2]
    circle = patches.Circle((x, y), radius, color="#c83737ff")  # (center_x, center_y), radius
    ax.add_patch(circle)
# hide ticks
ax.set_xticks([])
ax.set_yticks([])
# save the figure
if deterministic:
    fig_name += "_Det"
else:
    fig_name += "_Stoch"
fig.savefig("Figures//" + fig_name + ".png", dpi=300, format='png')
fig.savefig("Figures//" + fig_name + ".svg")


