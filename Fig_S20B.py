import numpy as np
import matplotlib.pyplot as plt

# Generate x values over the range [0, 10]
x = np.linspace(0, 10, 1000)

# Define the function: 1 + UnitStep(x-1.045) + UnitStep(x-3.175) + UnitStep(x-6.215)
# np.heaviside with 1 as the second argument matches Mathematica's UnitStep behavior
y = (1 + 
     np.heaviside(x - 1.045, 1) + 
     np.heaviside(x - 3.175, 1) + 
     np.heaviside(x - 6.215, 1))

# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the function with blue color and thickness equivalent to Mathematica's Thickness[0.008]
ax.plot(x, y, color='blue', linewidth=2.5, solid_capstyle='butt')

# Set the plot range (equivalent to PlotRange -> {0, 4.5})
ax.set_ylim(0, 4.5)
ax.set_xlim(0, 10)

# Set frame labels (equivalent to FrameLabel)
ax.set_xlabel(r'$\kappa$', fontsize=12, color='black')  # \[Kappa] rendered as ?
ax.set_ylabel('Leading Fourier Mode (n)', fontsize=12, color='black')

# Configure frame visibility: Frame -> {True, True, False, False}
# Shows bottom and left spines only, hides top and right
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set frame style: Black with thickness equivalent to Thickness[0.005]
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color('black')
    ax.spines[spine].set_linewidth(1.5)

# Ensure labels and ticks are black (equivalent to LabelStyle -> Directive[Black, Medium])
ax.tick_params(axis='both', colors='black', labelsize=10)

plt.tight_layout()
plt.show()