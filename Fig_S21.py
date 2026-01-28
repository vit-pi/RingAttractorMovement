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
fig, axs = plt.subplots(1, 3, figsize=(3*cell_size, cell_size), sharey=True)

# Define parameters
kappa = 1
s0 = 1
delta_thetas = [np.pi / 6, np.pi/2, np.pi]
theta = np.linspace(-np.pi, np.pi, 500)


# Define the two functions
def s(theta, delta_theta):
    term1 = s0 * np.exp(kappa * (np.cos(theta - delta_theta / 2) - 1))
    term2 = s0 * np.exp(kappa * (np.cos(theta + delta_theta / 2) - 1))
    term3 = s0 ** 2 * np.exp(2 * kappa * (np.cos(theta) * np.cos(delta_theta / 2) - 1))
    return term1 + term2 - term3


def s_tilde(theta, delta_theta):
    return s0 ** 2 * np.exp(2 * kappa * (np.cos(theta) * np.cos(delta_theta / 2) - 1))


# Plotting
for ax, delta_theta in zip(axs, delta_thetas):
    s_vals = s(theta, delta_theta)
    s_tilde_vals = s_tilde(theta, delta_theta)

    ax.plot(theta, s_vals, color='grey', label=r'additive')
    ax.plot(theta, s_tilde_vals, color='purple', label=r'multiplicative')
    ax.fill_between(theta, s_vals, color='grey', alpha=0.3)
    ax.fill_between(theta, s_tilde_vals, color='purple', alpha=0.3)

    ax.set_title(r'$\Delta\theta = {:.0f}^\circ$'.format(np.degrees(delta_theta)))
    ax.set_xlabel(r'angle $\theta$')
    ax.set_xlim([-np.pi, np.pi])
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    ax.set_ylim([0, 1.05])
axs[0].set_ylabel(r'stimulation $s(\theta), \tilde{s}(\theta)$')
axs[0].legend(loc='upper center')
fig.tight_layout()

###
# SAVE
###
fig.savefig("Figures/Fig_ModelStimulationAddMult.png", dpi=300)
fig.savefig("Figures/Fig_ModelStimulationAddMult.svg")
plt.show()