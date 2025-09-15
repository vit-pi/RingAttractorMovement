import numpy as np
import matplotlib.pyplot as plt

# Values for s0 and kappa
s0_values = [0.2, 0.8]
kappa_values = [0, 0.5]

# Theta values from -pi to pi
theta = np.linspace(-np.pi, np.pi, 1000)

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(2, 2))
axes = axes.ravel()  # Flatten the 2D array of axes for easier indexing

for i, s0 in enumerate(s0_values):
    for j, kappa in enumerate(kappa_values):
        # Calculate the function values
        y = s0 * np.exp(kappa * (np.cos(theta) - 1))

        # Select the current axis
        ax = axes[i * 2 + j]

        # Plot the function in black
        ax.plot(theta, y, color='black')

        # Fill the area below the function with grey shading
        ax.fill_between(theta, y, color='grey', alpha=0.5)

        # Adjust the limits and labels
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([0, 1])
        ax.set_xticks([])
        ax.set_yticks([])

# Adjust layout
plt.tight_layout()

###
# SAVE
###
fig.savefig("Figures/Fig_Transition_Phases_Inset.png", dpi=300)
fig.savefig("Figures/Fig_Transition_Phases_Inset.svg")
plt.show()