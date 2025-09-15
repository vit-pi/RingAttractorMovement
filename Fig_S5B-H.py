# IMPORTS
import matplotlib.pyplot as plt
import BrainMove as bm

# VARIABLES
interaction = [0,2,1]
T = 0.5
S = 1
strength_diff = [[0,0,0,0.05],[0.1,0.02,0.15,0.1]]
theta = [[2.1,2.15,2.24,2.1],[2.1,2.145,2.13,2.22]]
s1 = 0
fig_name = "TwoTargetsPhasePort"

# DO THE PLOT
# prepare figure
fig,axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 5), layout="constrained")
# make plots
for row in range(2):
    for col in range(4):
        targets = bm.TwoTargets(interaction, T, [1-s1, 1-s1-strength_diff[row][col]], S, theta[row][col])
        targets.plot_phase_portrait(axs[row][col])
# save the figure
fig.savefig("Figures//" + fig_name + ".png", dpi=300, format='png')
fig.savefig("Figures//" + fig_name + ".svg")