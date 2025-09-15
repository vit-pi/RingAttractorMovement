# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc_sp

# PREPARE PLOTTING DATA
# Lambert W
branches = 5
alphas = [-2, -0.5, 0.5, 2.5]
# Real roots
tau = 0.7#0.8
alpha_real = 3
# J modes
angle_num = 50
J = 2
nus = [0.5, 0.75, 1]
# Phase diag
alpha_max = 6
tau_max = 6

# PREPARE PLOT
fig = plt.figure(figsize=(10,4.5), layout="constrained")
# crate axes
gs = fig.add_gridspec(2, 4)
ax_tiny_a = fig.add_subplot(gs[0, 0])
ax_low_a = fig.add_subplot(gs[0, 1])
ax_medium_a = fig.add_subplot(gs[0, 2])
ax_high_a = fig.add_subplot(gs[0, 3])
ax_real = fig.add_subplot(gs[1, 0])
ax_J = fig.add_subplot(gs[1, 1])
ax_diag = fig.add_subplot(gs[1, 2:])

# FUNCTION TO PLOT LAMBERT FUNCTIONS INTO GIVEN AXES
def plot_defining_equation(ax,tau,alpha):
    num = 50
    sigmas_low = np.linspace(-5,-(tau+1)/tau,num=num)
    sigmas_high = np.linspace(-(tau+1)/tau, 1, num=num)
    fun_low = np.exp(tau*sigmas_low)*(sigmas_low+1)
    fun_high = np.exp(tau * sigmas_high) * (sigmas_high + 1)
    ax.plot(sigmas_low, fun_low,color='tab:red')
    ax.plot(sigmas_high, fun_high, color="#B79000")
    ax.set_xlim([-5, 1])
    #ax.set_xticks([-2,-1,0,1,2])
    ax.axhline(y=alpha, color='black')
    ax.axhline(y=1, color='black', linestyle=':')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel("real eigenvalue $\\sigma_k$")
    ax.set_ylabel("function $e^{\\sigma_k \\tau}(\\sigma_k+1)$")
    ax.set_yticks([0, 1, alpha])
    ax.set_yticklabels([0, 1, "$\\alpha_k$"])
    #ax.set_yticks([-np.exp(-tau-1)/tau,0,1,alpha])
    #ax.set_yticklabels(["$e^{-\\tau-1}/\\tau$", 0, 1, "$\\alpha_k$"])
    ax.set_xticks([-(tau+1)/tau, -1, 0])
    ax.set_xticklabels(["-1-1/$\\tau$", -1, 0])
    ax.set_ylim(-0.5,5)

# FUNCTION TO PLOT LAMBERT FUNCTIONS INTO GIVEN AXES
def plot_lambert(ax,branches,alpha):
    colors = ["#B79000", 'tab:red', "#DF4848", "#E86A6A", "#F08B8B", "#F8ADAD"]
    num = 100
    tau_max = 7
    taus = np.linspace(0,tau_max,num=num)
    lambdas = np.zeros((branches,num))
    tau_crit = 0
    for branch in range(branches):
        for n in range(num):
            if n == 0:
                if branch == 0:
                    lambdas[branch, n] = -1+alpha
                else:
                    lambdas[branch, n] = 'NaN'
            else:
                lambdas[branch, n] = np.real(-1+sc_sp.lambertw(alpha*taus[n]*np.exp(taus[n]), k=branch)/taus[n])
                if alpha>1 and branch == 1 and lambdas[branch, n]*lambdas[branch, n-1] < 0:
                    tau_crit = (n-1/2)*tau_max/(num-1)
                if alpha<-1 and branch == 0 and lambdas[branch, n]*lambdas[branch, n-1] < 0:
                    tau_crit = (n-1)*tau_max/(num-1)
        ax.plot(taus, lambdas[branch],color=colors[branch], label="l="+str(branch))
    ax.set_xlim([0, tau_max])
    if alpha>1:
        ax.set_ylim([-2, 2])
        ax.set_xticks([i for i in range(tau_max+1)])
        #ax.set_xticks([0, tau_crit,tau_max])
        #ax.set_xticklabels(["0", "$\\tau_c$",tau_max])
    elif alpha>-1:
        ax.set_ylim([-3, 1])
        ax.set_xticks([i for i in range(tau_max + 1)])
        #ax.set_xticks([0,tau_max])
        #ax.set_xticklabels(["0",tau_max])
    else:
        ax.set_ylim([-5, 1])
        ax.set_xticks([i for i in range(tau_max + 1)])
        #ax.set_xticks([0, tau_crit,tau_max])
        #ax.set_xticklabels(["0", "$\\tau_c$",tau_max])
    ax.hlines(y=0, xmin=0, xmax=tau_max, color='black')
    ax.set_xlabel("neurotransmission delay $\\tau$")
    ax.set_ylabel("eigenvalue Re$(\\lambda_k^l)$")
    #ax.set_yticks([0,alpha-1])
    #ax.set_yticklabels(["0","$\\alpha_k-1$"])
    ax.legend()

# FUNCTION TO PLOT FOURIER MODES OF J
# function J
def find_J(theta, nu):
    return J * np.cos(np.pi * (np.abs(theta) / np.pi) ** nu)

# Fourier modes of J
def find_Jk(k, nu):
    angle_step = 2*np.pi/angle_num
    J_k = 0
    for ang in range(angle_num):
        angle = ang * angle_step - np.pi
        J_k += find_J(angle, nu) * np.cos(k * angle)
    return J_k

def plot_Jk(ax, nus):
    num = 10
    # create plotting data
    ks = np.linspace(0,num-1,num=num)
    Jks = np.zeros((len(nus),num))
    color = ["dimgrey", "darkgrey", "lightgrey"]
    marker = ["o", "^", "s"]
    for nu in range(len(nus)):
        for k in range(num):
            Jks[nu, k] = find_Jk(k, nus[nu])
        # make plot
        ax.scatter(ks, Jks[nu], label="$\\nu=$" + str(nus[nu]), color=color[nu], marker=marker[nu],zorder=10)
    # labels
    ax.set_xlabel("mode $k$")
    ax.set_ylabel("Fourier transform $J_k$")
    ax.legend()
    # limits
    ax.set_xlim([-0.5, num-0.5])
    ax.set_xticks([i for i in range(num)])

# FUNCTION TO PLOT PHASE DIAGRAM
def plot_phase_diag(ax):
    # set labels
    ax.set_xlabel("excitation $\\alpha_k$")
    ax.set_ylabel("neurotransmission delay $\\tau$")
    # set limits
    ax.set_xlim([-alpha_max, alpha_max])
    ax.set_ylim([0, tau_max])
    # draw dividing curves
    ax.axvline(x=1, color="#B79000")
    ax.axvline(x=0, color='black', linestyle=":", linewidth=1)
    num = 100
    l_max = 5
    alphas_pos = np.linspace(1 - (alpha_max - 1) / num, alpha_max, num=num)
    tau_c_pos = np.zeros((l_max,num))
    alphas_neg = np.linspace(-alpha_max, -1 + (alpha_max - 1) / num, num=num)
    tau_c_neg = np.zeros((l_max,num))
    colors_pos = ["tab:red", "#DF4848", "#E86A6A", "#F08B8B", "#F8ADAD"]
    colors_neg = ["#B79000", "tab:red", "#DF4848", "#E86A6A", "#F08B8B", "#F8ADAD"]
    for l in range(l_max):
        for n in range(num):
            if n == 0:
                tau_c_pos[l][n] = np.inf
                tau_c_neg[l][num-n-1] = np.inf
            else:
                tau_c_pos[l][n] = (2 * np.pi*(l+1) - np.arccos(1 / alphas_pos[n])) / np.sqrt(alphas_pos[n] ** 2 - 1)
                tau_c_neg[l][num-n-1] = (2 * np.pi*(l+1) - np.arccos(1 / alphas_neg[num-n-1])) / np.sqrt(alphas_neg[num-n-1] ** 2 - 1)
        ax.plot(alphas_pos, tau_c_pos[l], color=colors_pos[l])
        ax.plot(alphas_neg, tau_c_neg[l], color=colors_neg[l])
    # fill area between curves
    fill_colors = ["#EC7063", "#F18D82", "#F6AAA1", "#FBC7C0", "#FFE4DF"]
    ax.fill_between(alphas_pos, tau_c_pos[0], color=fill_colors[0])
    for l in range(l_max-1):
        ax.fill_between(alphas_pos, tau_c_pos[l], tau_c_pos[l + 1], color=fill_colors[l + 1])
        ax.fill_between(alphas_neg, tau_c_neg[l], tau_c_neg[l + 1], color=fill_colors[l])
    ax.fill_between([alphas_neg[-1] - 20 / num, 1], [tau_max, tau_max], color="#FDE27D")
    ax.fill_between(alphas_neg, tau_c_neg[0], color="#FDE27D")
# MAKE PLOT
plot_defining_equation(ax_real,tau,alpha_real)
plot_lambert(ax_tiny_a, branches, alphas[0])
plot_lambert(ax_low_a, branches, alphas[1])
plot_lambert(ax_medium_a, branches, alphas[2])
plot_lambert(ax_high_a,branches,alphas[3])
plot_Jk(ax_J,nus)
plot_phase_diag(ax_diag)
# add titles
fig.savefig("Figures/Fig_LambertFourier.svg")
fig.savefig("Figures/Fig_LambertFourier.png")
plt.show()