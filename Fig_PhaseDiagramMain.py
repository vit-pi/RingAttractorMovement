# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import BrainMove as bm

# PREPARE PLOTTING DATA
tau_max = 20
kernel = [0, 10, 0.5]
T = 1
angle_num = 64

# PREPARE PLOT
fig = plt.figure(figsize=(3.2,3), layout="constrained")
# create axes
gs = fig.add_gridspec(1, 1)
ax_diag = fig.add_subplot(gs[0,0])

# MAKE PLOT
# set labels
ax_diag.set_ylabel("intrinsic stimulation $S$")
ax_diag.set_xlabel("neurotransmission delay $\\tau$")
# set limits
ax_diag.set_xlim(0, tau_max)
ax_diag.set_ylim(0, 1)
# find the dividing curves for each l
num = 100
l_max = 5
ss = np.linspace(0,1,num=num)
alphas = np.zeros(num)
tau_c = np.zeros((l_max,num))
colors = ["#D62728", "#DF4848", "#E86A6A", "#F08B8B", "#F8ADAD"]
for n in range(num):
    beh_prop = bm.BehProp(kernel=kernel,S=ss[n],T=T,angle_num=angle_num)
    alphas[n] = beh_prop.alpha(1)
    for l in range(l_max):
        if alphas[n] <= 1:
            tau_c[l][n] = np.inf
        else:
            tau_c[l][n] = (2*np.pi*(l+1)-np.arccos(1/alphas[n]))/np.sqrt(alphas[n]**2-1)
# draw dividing curves
for l in range(l_max):
    ax_diag.plot(tau_c[l],ss,color=colors[l])
# find critical s
i=0
while alphas[i]<=1:
    i+=1
s_crit = ss[i:i+1][0]
ax_diag.axhline(y=s_crit, color="#B79000")
# fill area between curves
fill_colors = ["#EC7063", "#F18D82", "#F6AAA1", "#FBC7C0", "#FFE4DF"]
ax_diag.fill_betweenx(ss[i:], tau_c[0][i:], color=fill_colors[0])
for l in range(l_max-1):
    ax_diag.fill_betweenx(ss[i:], tau_c[l][i:],tau_c[l+1][i:], color=fill_colors[l+1])
ax_diag.fill_between([0,tau_max], [s_crit,s_crit], color="#FDE27D")
# save figure
fig.savefig("Figures/PhaseDiag.svg")
fig.savefig("Figures/PhaseDiag.png")
plt.show()