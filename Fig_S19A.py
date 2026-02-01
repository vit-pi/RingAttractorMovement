import numpy as np
import matplotlib.pyplot as plt

# Define the data points (Pnts)
# Note: 20/(2*Pi) is the common factor used in most points
factor = 20 / (2 * np.pi)

Pnts = [
    [0.1 * np.pi, 17.1 * factor],
    [0.15 * np.pi, 6.2],                    # No factor multiplier on 6.2 (as in original)
    [0.2 * np.pi, 1.18 * factor],
    [0.3 * np.pi, 0.52 * factor],
    [0.4 * np.pi, 0.37 * factor],
    [0.5 * np.pi, 0.33 * factor],
    [0.55 * np.pi, 0.0 + 256 * factor],     # 0.0 + preserves original syntax
    [0.6 * np.pi, 0.0 + 698 * factor]
]

# Convert to numpy arrays for plotting
x_vals = np.array([p[0] for p in Pnts])
y_vals = np.array([p[1] for p in Pnts])

# Color definitions (RGBColor values from Mathematica)
salmon_color = (250/256, 128/256, 114/256)  # RGBColor[250/256, 128/256, 114/256]
yellow_color = (250/256, 218/256, 94/256)   # RGBColor[250/256, 218/256, 94/256]

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Set plot range: {{0, Pi}, {0, 20}}
ax.set_xlim(0, np.pi)
ax.set_ylim(0, 20)

# Horiz1: Plot[20, {x, 1.6, Pi}, Filling -> Bottom, FillingStyle -> yellow]
x_horiz1 = np.linspace(1.6, np.pi, 100)
y_horiz1 = np.full_like(x_horiz1, 20)
ax.fill_between(x_horiz1, y_horiz1, 0, color=yellow_color, zorder=1)
ax.plot(x_horiz1, y_horiz1, color=yellow_color, linewidth=1.0, zorder=2)

# Horiz2: Plot[20, {x, 0, 0.35}, Filling -> Bottom, FillingStyle -> yellow]
x_horiz2 = np.linspace(0, 0.35, 100)
y_horiz2 = np.full_like(x_horiz2, 20)
ax.fill_between(x_horiz2, y_horiz2, 0, color=yellow_color, zorder=1)
ax.plot(x_horiz2, y_horiz2, color=yellow_color, linewidth=1.0, zorder=2)

# Bottompart: ListLinePlot with Filling -> Bottom (to y=0) with yellow
ax.fill_between(x_vals, y_vals, 0, color=yellow_color, zorder=3)

# Upperpart: ListLinePlot with Filling -> Top (to y=20) with salmon
ax.fill_between(x_vals, y_vals, 20, color=salmon_color, zorder=4)

# Plot the main line from Pnts
# PlotStyle: Directive[Black, Thickness[0.003]] -> approximated as linewidth=0.5
# Note: Mathematica's Thickness is relative; 0.003 ? 0.5 points in matplotlib
ax.plot(x_vals, y_vals, color='black', linewidth=0.5, zorder=5)

# FrameLabel conversion (Mathematica subscript boxes to LaTeX)
# " \!\(\*SubscriptBox[\(\[Theta]\), \(0\)]\)" -> r' $\theta_0$'
# "\!\(\*SubscriptBox[\(J\), \(-\)]\)/T" -> r'$J_-/T$'
ax.set_xlabel(r' $\theta_0$', fontsize=12, fontweight='bold')
ax.set_ylabel(r'$J_-/T$', fontsize=12, fontweight='bold')

# Frame settings: Frame -> {True, True, False, False}
# Only bottom and left axes visible
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# FrameStyle -> Medium (set linewidth of visible spines)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)

# Ensure tight layout
plt.tight_layout()

# Display the plot (equivalent to showing FirstPlotBumpForStep)
plt.show()