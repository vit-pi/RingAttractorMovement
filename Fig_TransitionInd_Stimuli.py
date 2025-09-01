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
fig,axs = plt.subplots(nrows=3,figsize=(cell_size, 1.7*cell_size), layout="constrained")

# Plot the function
# angles
theta=np.linspace(-np.pi,np.pi,num=100)
# variables
kappa = 10
s = 1
s_alternative = 0.8
kappa_alternative = 20
# curves
exc=s*np.exp(kappa*(np.cos(theta)-1))
exc_s=s_alternative*np.exp(kappa*(np.cos(theta)-1))
exc_kappa=s*np.exp(kappa_alternative*(np.cos(theta)-1))
for i in range(3):
    # plot the curves
    axs[i].plot(theta, exc, color="black", linestyle="--")
    # Fill the regions
    # axs[i].fill_between(theta, 0, exc, color='gray',alpha=0.5)
    # Set x and y limits
    axs[i].set_xlim(-np.pi, np.pi)
    axs[i].set_ylim(0, 1.05)
    # Set the ticks
    axs[i].set_xticks([-np.pi, 0, np.pi])
    axs[i].set_yticks([0, 1])
    # Label the ticks
    axs[i].set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    axs[i].set_yticklabels([r'$0$', r'$1$'])
    # Labels
    axs[i].set_xlabel("angle from cue $\\Delta \\theta$")
    axs[i].set_ylabel("stimulation $s(\\Delta \\theta,t)$")
# plot the alternatives
axs[1].plot(theta, exc_s, color="black", linestyle=":")
axs[2].plot(theta, exc_kappa, color="black", linestyle=":")
# fill the alternatives
# axs[1].fill_between(theta, 0, exc_s, color='gray',alpha=0.5)
# axs[2].fill_between(theta, 0, exc_kappa, color='gray',alpha=0.5)

###
# SAVE
###
fig.savefig("Figures/Fig_TransitionInd_Stimuli.png", dpi=300)
fig.savefig("Figures/Fig_TransitionInd_Stimuli.svg")
plt.show()