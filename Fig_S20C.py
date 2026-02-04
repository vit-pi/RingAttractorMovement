import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Bump values
bump1 = np.array([0.94,0.94,0.93,0.91,0.89,0.84,0.76,0.64,0.48,0.33,0.22,0.15,0.11,0.09,0.078,0.072,
                  0.071,0.072,0.078,0.09,0.11,0.15,0.22,0.33,0.48,0.64,0.76,0.84,0.89,0.91,0.93,0.94])
bump2 = np.array([1.,1.,1.,0.97,0.5,0.035,0.0023,0.00038,0.0002,0.00038,0.0023,0.035,0.5,0.97,1.,1.,
                  1.,1.,1.,0.97,0.5,0.035,0.0023,0.00038,0.0002,0.00038,0.0023,0.035,0.5,0.97,1.,1.])
bump3 = np.array([1.,1.,1.,0.00023,1.6e-6,1.5e-6,1.9e-6,2.8e-6,0.0002,1.,1.,1.,1.,1.,0.000012,7.2e-6,
                  6.7e-6,7.2e-6,0.000012,1.,1.,1.,1.,1.,0.0002,2.8e-6,1.9e-6,1.5e-6,1.6e-6,0.00023,1.,1.])

all_bumps = [bump1, bump2, bump3]
titles = ['Bump 1', 'Bump 2', 'Bump 3']

for bump, title in zip(all_bumps, titles):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))
    cmap = get_cmap('Reds')
    angles = np.linspace(0, 2 * np.pi, len(bump), endpoint=False)
    
    bars = ax.bar(angles, np.ones(len(bump)), width=2*np.pi/len(bump), bottom=1,
                  color=cmap(bump), edgecolor='black')

    ax.set_xticklabels([])  # Remove angle labels
    ax.set_yticklabels([])  # Remove radial labels
    ax.grid(False)
    # ax.set_title(title)  # Optional title

    plt.tight_layout()
    plt.show()

