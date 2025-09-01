# IMPORTS
import numpy as np
import matplotlib.pyplot as plt

# PREPARE PLOTTING DATA (normalise by setting T=1)
S = np.exp(-np.array([5, 7, 5, 2, 1.5, -5]))
J0 = [10, 10, 10, 10, 3, -10]

# PREPARE PLOT
fig = plt.figure(figsize=(10,4.5), layout="constrained")
# crate axes
gs = fig.add_gridspec(2, 4)
ax_legend = fig.add_subplot(gs[0, 0])
axs = []
axs.append(fig.add_subplot(gs[0, 1]))
axs.append(fig.add_subplot(gs[0, 2]))
axs.append(fig.add_subplot(gs[0, 3]))
axs.append(fig.add_subplot(gs[1, 0]))
axs.append(fig.add_subplot(gs[1, 1]))
ax_phase = fig.add_subplot(gs[1, 2:])

# FUNCTION TO PLOT LAMBERT FUNCTIONS INTO GIVEN AXES
def plot_sigmoid(ax,S,J):
    # plot
    n = np.linspace(0,1,num=100)
    sigmoid = S/(S+np.exp(-J*n))
    ax.plot(n,sigmoid, color='black', linewidth=1)
    # limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    # labels
    ax.set_xlabel("$n$")
    ax.set_ylabel("$\\sigma$")

def plot_line(ax,eta,J):
    ax.plot([0,1],[0,1], color='black', linewidth=1)

def plot_phase_diag(ax):
    # set labels
    ax.set_xlabel("$J_0/T$")
    ax.set_ylabel("$S$")
    # set limits
    min = -5
    max = 10
    ax.set_xlim([min, max])
    ax.set_ylim([1e-4, 10])
    ax.set_xticks([-5,0, 5, 10])
    # ax.set_yticks([0,0.5,1])
    # draw dividing curves
    ax.axvline(x=0, color='black', linestyle=":", linewidth=1)
    J0 = np.linspace(4,max)
    x_plus = 0.5 + 0.5*np.sqrt(1-4/J0)
    x_minus = 0.5 - 0.5 * np.sqrt(1 - 4 / J0)
    S_plus = np.exp(-J0*x_plus-np.log(x_minus/x_plus))
    S_minus = np.exp(-J0 * x_minus - np.log(x_plus / x_minus))
    ax.plot(J0, S_plus, color='black', linewidth=1)
    ax.plot(J0, S_minus, color='black', linewidth=1)
    # fill area between curves
    ax.fill_between(J0,S_minus, S_plus, color='grey')
    # set scale to logarithmic
    ax.set_yscale('log')
# MAKE PLOT
plot_sigmoid(ax_legend,S[0],J0[0])
for frame in range(5):
    plot_line(axs[frame], S[frame+1], J0[frame+1])
    plot_sigmoid(axs[frame], S[frame+1], J0[frame+1])
plot_phase_diag(ax_phase)
# SAVE FIGURE
fig.savefig("Figures/Fig_HomogSteadySt.svg")
fig.savefig("Figures/Fig_HomogSteadySt.png")
plt.show()