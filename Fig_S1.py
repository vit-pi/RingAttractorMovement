###
# IMPORT
###
import numpy as np
import matplotlib.pyplot as plt

###
# FIGURE
###
# create figure
cell_size = 2.5
fig, axs = plt.subplots(1, 2, figsize=(2*cell_size, cell_size))

# Define parameters
d_c_fixed = 10         # Fixed d_c value
rho_c_fixed = 2     # Fixed rho_c value
rho_c_values = np.linspace(0.01, d_c_fixed, 500)   # Avoid division by zero
d_c_values = np.linspace(rho_c_fixed, d_c_fixed, 500)    # Avoid division by zero

# Define function
def kappa_c(rho_c, d_c):
    with np.errstate(divide='ignore', invalid='ignore'):  # Handle any invalid calculations
        result = (np.pi / (2 * np.arcsin(rho_c / d_c)**2)) - 1 / (2 * np.pi)
        result[np.isnan(result) | np.isinf(result)] = 0  # Replace invalid results
        return result

# Compute values for panel 1
rho_c = rho_c_values  # x-axis

kappa_c_panel1 = kappa_c(rho_c, d_c_fixed)

# Compute values for panel 2
d_c = d_c_values      # x-axis

kappa_c_panel2 = kappa_c(rho_c_fixed, d_c)

# Plotting Panel 1: kappa_c vs rho_c
axs[0].plot(rho_c, kappa_c_panel1, color='black', label='$d_c = 1$')
axs[0].set_xlim(0, d_c_fixed)
axs[0].set_ylim(0, 40)
axs[0].set_ylabel('parameter $\\kappa_c$')
axs[0].set_xlabel('radius $\\rho_c$')

# Plotting Panel 2: kappa_c vs d_c
axs[1].plot(d_c, kappa_c_panel2, color='black', label='$\\rho_c = 0.5$')
axs[1].set_xlim(0, d_c_fixed)
axs[1].set_ylim(0, 40)
axs[1].set_ylabel('parameter $\\kappa_c$')
axs[1].set_xlabel('distance $d_c$')

fig.tight_layout()

###
# SAVE
###
fig.savefig("Figures/Fig_ModelStimulationOverview.png", dpi=300)
fig.savefig("Figures/Fig_ModelStimulationOverview.svg")
plt.show()