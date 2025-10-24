import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import matplotlib as mpl

mpl.rcParams.update({
    # font
    "font.family"     : "sans-serif",
    "font.sans-serif" : ["Helvetica"],   # fall-back handled automatically
    "font.size"       : 20,              # <--- main control for larger text
    "axes.titlesize"  : 20,              # axes title size
    "axes.labelsize"  : 20,              # x/y label size
    "xtick.labelsize" : 20,              # x tick label size
    "ytick.labelsize" : 20,              # y tick label size
    "legend.fontsize" : 20, 
    "text.color"      : "black",
    # axes & ticks
    "axes.labelcolor" : "black",
    "axes.edgecolor"  : "black",
    "xtick.color"     : "black",
    "ytick.color"     : "black",
    "axes.facecolor"  : "white",
    "figure.facecolor": "white",
    # grid (light grey, thin, dashed - similar to ggplot2::theme_bw)
    "axes.grid"       : False,
    "grid.color"      : "0.7",
    "grid.linestyle"  : "--",
    "grid.linewidth"  : 0.1,
})

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <values.txt>")
    sys.exit(1)

filename = sys.argv[1]

# Read values from file (one per line)
with open(filename, "r") as f:
    data = [float(line.strip()) for line in f if line.strip()]

arr = np.array(data)

# Histogram
plt.hist(arr, bins=20, edgecolor="black", alpha=0.7, density=True, label="Histogram")

# Kernel Density Estimate (smooth curve)
kde = gaussian_kde(arr)
x_vals = np.linspace(arr.min(), arr.max(), 200)
plt.plot(x_vals, kde(x_vals), color="red", lw=2, label="KDE")

# Labels and title
plt.xlabel("Value")
plt.ylabel("Density")
plt.title(f"Distribution of Values from {filename}")
plt.legend()
plt.savefig('branch_relevant.pdf', bbox_inches='tight')
# Show plot
plt.show()
