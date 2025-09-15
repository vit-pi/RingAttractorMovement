# IMPORTS
import BrainMove as bm

# PREPARE PLOTTING DATA
kernel = [0,10,0.5]#[0, 2, 0.5]
S = 1#0.3
T = 1#0.2
# MAKE THE PLOT
bm.plot_alpha_dependence(kernel,S,T)