# IMPORTS
import BrainMove as bm

# VARIABLES
interaction = [0,2,1]
T = 0.5
s1 = 0
s2_max = 0.15
S = 1
fig_name = "TwoTargetsPhaseDiag"

# DO THE PLOT
bm.plot_phase_diagram(interaction,T,s1,s2_max,S,fig_name)