###
# IMPORT
###
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

###
# FIGURE
###
# create figure
cell_size = 2.5
fig,axs = plt.subplots(nrows=4,figsize=(0.9*cell_size, cell_size), layout="constrained")

###
# COLORBARS
###
# Define the colormap and normalization
yellows = LinearSegmentedColormap.from_list("Yellows", [(0.0,(1,1,0.8)), (0.2,(0.8, 0.6, 0)),(1.0,(0.4, 0.3, 0.00))])
cmaps = [plt.get_cmap('Blues'),plt.get_cmap('Reds'),yellows,plt.get_cmap('Greys')]
norm = mcolors.Normalize(vmin=0, vmax=1)
# Create a ScalarMappable object for the colorbar
for i in range(4):
    sm = cm.ScalarMappable(cmap=cmaps[i], norm=norm)
    sm.set_array([])  # Dummy array for the colorbar
    # Add the colorbar to the given axes (ax)
    cbar = plt.colorbar(sm, cax=axs[i], orientation="horizontal")
    cbar.set_label('neural activity $n(\\theta,t)$')

###
# SAVE
###
fig.savefig("Figures/Fig_Colorbars.png", dpi=300)
fig.savefig("Figures/Fig_Colorbars.svg")
plt.show()