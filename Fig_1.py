###
# IMPORT
###
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import scipy.special as special
from matplotlib import font_manager

###
# FIGURE
###
# create figure
cell_size = 2.5
fig = plt.figure(figsize=(cell_size, cell_size*2), layout="constrained")
# crate axes
gs = fig.add_gridspec(4*5, 1)
ax_direction = fig.add_subplot(gs[0:5])
ax_cue = fig.add_subplot(gs[5:10])
ax_speed = fig.add_subplot(gs[10:15])
ax_colour = fig.add_subplot(gs[15:16])

###
# DIRECTION NEURONS
###
# Plot the function
theta=np.linspace(-np.pi,np.pi,num=100)
J_vals=np.cos(np.pi*(np.abs(theta)/np.pi)**0.5)
#norm = special.iv(0,1)*np.exp(-1)
#J_vals=(np.exp((np.cos(theta)-1))-norm)/(1-norm)
ax_direction.plot(theta, J_vals, color='black')
# Fill the regions
ax_direction.fill_between(theta, 0, 1, color='#ec7063ff')  # Above the x-axis
ax_direction.fill_between(theta, -1, 0, color='#FDE27D')   # Below the x-axis
# Set x and y limits
ax_direction.set_xlim(-np.pi, np.pi)
ax_direction.set_ylim(-1, 1)
# Set the ticks
ax_direction.set_xticks([-np.pi, 0, np.pi])
ax_direction.set_yticks([-1,0, 1])
# Label the ticks
ax_direction.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
ax_direction.set_yticklabels([r'$-J$',r'$0$', r'$J$'])
# Labels
ax_direction.set_xlabel("angular difference $\\Delta\\theta$")
ax_direction.set_ylabel("interaction $J(\\Delta\\theta)$")

###
# CUE NEURONS
###
# Plot the function
theta=np.linspace(-np.pi,np.pi,num=100)
kappa = 10
phi = np.pi
exc0=np.exp(kappa*(np.cos(theta-phi/2)-1))
exc1=np.exp(kappa*(np.cos(theta+phi/2)-1))
exc_tot=1-(1-exc0)*(1-exc1)
ax_cue.plot(theta, exc0, color="black")
# ax_cue.plot(theta, exc_tot, color="black")
# ax_cue.plot(theta, exc0, color="gray",linestyle=":")#color="#D62728")
# ax_cue.plot(theta, exc1, color="gray",linestyle=":")#color="#B79000")
# Fill the regions
# ax_cue.fill_between(theta, 0, exc_tot, color='lightgray')
ax_cue.fill_between(theta, 0, exc0, color='lightgray')
# Set x and y limits
ax_cue.set_xlim(-np.pi, np.pi)
ax_cue.set_ylim(0, 1.05)
# Set the ticks
ax_cue.set_xticks([-np.pi, 0, np.pi])
ax_cue.set_yticks([0, 1])
# Label the ticks
ax_cue.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
ax_cue.set_yticklabels([r'$0$', r'$1$'])
# Labels
ax_cue.set_xlabel("angle $\\theta$")
ax_cue.set_ylabel("stimulation $s(\\theta,t)$")

###
# SPEED NEURONS
###
# Plot the function
ang_speed=np.sin(theta)
ax_speed.plot(theta, ang_speed, color='black')
# Set x and y limits
ax_speed.set_xlim(-np.pi, np.pi)
ax_speed.set_ylim(-1.1, 1.1)
# Set the ticks
ax_speed.set_xticks([-np.pi, 0, np.pi])
ax_speed.set_yticks([-1, 0, 1])
# Label the ticks
ax_speed.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
ax_speed.set_yticklabels([r'$-2\omega_0$', r'$0$', r'$2\omega_0$'])
# Labels
ax_speed.set_xlabel("angular difference $\\Delta\\phi$")
ax_speed.set_ylabel("angular velocity")

###
# COLORBAR NEURONS
###
# Define the colormap and normalization
cmap = plt.get_cmap('Blues')
norm = mcolors.Normalize(vmin=0, vmax=1)
# Create a ScalarMappable object for the colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Dummy array for the colorbar

# Add the colorbar to the given axes (ax)
cbar = plt.colorbar(sm, cax=ax_colour, orientation="horizontal")
cbar.set_label('neural activity $n(\\theta,t)$')

### Sizes of ticklabels and axis labels adjusted to 80% of their default values ###
# Scale for size adjustments
scale_factor = 0.8

# Helper function to resolve font size names to numeric values
def resolve_fontsize(size):
    if isinstance(size, str):
        return font_manager.FontProperties(size=size).get_size_in_points()
    return size

# Adjust font sizes for all tick labels and axis labels globally
for ax in [ax_direction, ax_cue, ax_speed]:
    # Resolve tick label size and scale
    tick_label_size = resolve_fontsize(plt.rcParams['xtick.labelsize']) * scale_factor
    ax.tick_params(axis='both', labelsize=tick_label_size)

    # Resolve axis label sizes and scale
    x_label_size = resolve_fontsize(plt.rcParams['axes.labelsize']) * scale_factor
    y_label_size = resolve_fontsize(plt.rcParams['axes.labelsize']) * scale_factor
    ax.xaxis.label.set_size(x_label_size)
    ax.yaxis.label.set_size(y_label_size)

# Handle the colorbar axis (ax_colour)
tick_label_size = resolve_fontsize(plt.rcParams['xtick.labelsize']) * scale_factor
cbar.ax.tick_params(labelsize=tick_label_size)

x_label_size = resolve_fontsize(plt.rcParams['axes.labelsize']) * scale_factor
cbar.ax.xaxis.label.set_size(x_label_size)

###
# SAVE
###
fig.savefig("Figures/Fig_ModelOverview.png", dpi=300)
fig.savefig("Figures/Fig_ModelOverview.svg")
plt.show()