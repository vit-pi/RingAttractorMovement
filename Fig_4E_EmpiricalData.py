import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# File path and list of sheet names
fig_name = "TransientSearch_EmpiricalTrajectory"
file_path = "SourceData/EmpiricalTrajectories.xlsx"
sheet = "Data15"

# Create a figure and axes for subplots
fig, ax = plt.subplots(figsize=(4, 4))

# Read the trajectory data from the sheet
data = pd.read_excel(file_path, sheet_name=sheet)

# Define the colormap and generate 7 evenly spaced colors
cmap = LinearSegmentedColormap.from_list("BlueYellow", [(0.0,(0.031,0.239,0.494)),(1.0,(0.827, 0.659, 0.118))])
color = [cmap(i) for i in np.linspace(0, 1, 7)]  # Generate 7 colors

# Extract x and y positions
for i in range(7):
    x = data["Position X"+str(i+1)]
    y = data["Position Y"+str(i+1)]

    # Plot the trajectory
    ax.plot(x, y, label=sheet, color=color[i])  # Label with sheet name (e.g., Data1)
    ax.set_title(sheet)

    # Set equal aspect ratio for uniform scaling
    ax.set_aspect('equal', adjustable='box')

# Add grid for better readability
ax.grid(True)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
fig.savefig("Figures//" + fig_name + ".png", dpi=300, format='png')
fig.savefig("Figures//" + fig_name + ".svg")
plt.show()