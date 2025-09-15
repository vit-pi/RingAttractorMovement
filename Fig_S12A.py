# IMPORTS
import numpy as np
import matplotlib.pyplot as plt

# PREPARE PLOTTING DATA
alpha_max = 6
tau_max = 6

# PREPARE PLOT
fig = plt.figure(figsize=(2.5,2.5), layout="constrained")
# create axes
gs = fig.add_gridspec(1, 1)
ax_diag = fig.add_subplot(gs[0,0])

# MAKE PLOT
# set labels
ax_diag.set_xlabel("excitation $\\alpha$")
ax_diag.set_ylabel("neurotransmission delay $\\tau$")
# set limits
ax_diag.set_xlim([0, alpha_max])
ax_diag.set_ylim([0, tau_max])
# draw dividing curves
ax_diag.axvline(x=1, color="#B79000")
num = 100
l_max = 5#5
alphas = np.linspace(1-(alpha_max-1)/num, alpha_max, num=num)
tau_c = np.zeros((l_max,num))
colors = ["#D62728", "#DF4848", "#E86A6A", "#F08B8B", "#F8ADAD"]
for l in range(l_max):
    for n in range(num):
        if n == 0:
            tau_c[l][n] = np.inf
        else:
            tau_c[l][n] = (2*np.pi*(l+1)-np.arccos(1/alphas[n]))/np.sqrt(alphas[n]**2-1)
    ax_diag.plot(alphas, tau_c[l], color=colors[l])
# fill area between curves
fill_colors = ["#EC7063", "#F18D82", "#F6AAA1", "#FBC7C0", "#FFE4DF"]
ax_diag.fill_between(alphas, tau_c[0], color=fill_colors[0])
for l in range(l_max-1):
    ax_diag.fill_between(alphas, tau_c[l],tau_c[l+1], color=fill_colors[l+1])
ax_diag.fill_between([0,1], [tau_max,tau_max], color="#FDE27D")
# save figure
fig.savefig("Figures/PhaseDiag.svg")
fig.savefig("Figures/PhaseDiag.png")
plt.show()