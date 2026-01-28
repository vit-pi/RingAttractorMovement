import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 1))

# Remove the axis labels and ticks
ax.set_axis_off()

# Create a colormap
cmap = LinearSegmentedColormap.from_list("BlueYellow", [(0.0, (0.031, 0.239, 0.494)), (1.0, (0.827, 0.659, 0.118))])

# Define a gradient from 0 to 1
gradient = np.linspace(0, 1, 256).reshape(1, -1)

# Display the gradient with the colormap
im = ax.imshow(gradient, aspect="auto", cmap=cmap)

# Add a colorbar below the gradient
cbar = fig.colorbar(im, orientation='horizontal', ax=ax, pad=0.3)

# Add label for the colorbar
cbar.set_label("Plasma colormap")

# Show the plot
plt.show()
fig.savefig("Figures/Fig_Colorbar.svg")