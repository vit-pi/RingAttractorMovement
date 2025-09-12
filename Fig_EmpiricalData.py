import matplotlib.pyplot as plt
import pandas as pd

# File path and list of sheet names
fig_name = "EmpiricalTrajectories"
file_path = "SourceData/EmpiricalTrajectories.xlsx"
trajectory_sheets = ["Data1", "Data5", "Data3", "Data4", "Data1", "Data6", "Data7", "Data8", "Data11", "Data13"]

# Number of rows and columns for the subplot grid
num_rows, num_cols = 2, 5

# Create a figure and axes for subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))
axes = axes.flatten()  # Flatten into a 1D array for easier indexing

# Loop through the sheet names and plot trajectories
for i, sheet in enumerate(trajectory_sheets):
    if i >= len(axes):  # Ensure we don't exceed available subplot spaces
        break

    # Read the trajectory data from the sheet
    data = pd.read_excel(file_path, sheet_name=sheet)

    # Extract x and y positions
    x = data["Position X"]
    y = data["Position Y"]

    # Plot the trajectory
    axes[i].plot(x, y, label=sheet, color="#d4aa00ff")  # Label with sheet name (e.g., Data1)
    axes[i].set_title(sheet)

    # Set equal aspect ratio for uniform scaling
    axes[i].set_aspect('equal', adjustable='box')

    # Add grid for better readability
    axes[i].grid(True)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
fig.savefig("Figures//" + fig_name + ".png", dpi=300, format='png')
fig.savefig("Figures//" + fig_name + ".svg")
plt.show()