import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']


if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <values.txt>")
    sys.exit(1)

filename = sys.argv[1]

# Read values from file (one per line)
with open(filename, "r") as f:
    data = [float(line.strip()) for line in f if line.strip()]

arr = np.array(data)

# Histogram
plt.hist(arr, bins=10, edgecolor="black", alpha=0.7, density=True, label="Histogram")

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
