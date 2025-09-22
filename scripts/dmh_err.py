import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

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
# Load the data
df = pd.read_csv('dmh_benchmark.txt', sep='\t')

# Sort values by 'truth' just in case (optional, but keeps the line clean)
df = df.sort_values('truth')

plt.figure(figsize=(7, 5))

# Plot with both lines and points emphasized
plt.plot(df['truth'], df['relative_error'], marker='o', linestyle='-', linewidth=2, markersize=7, color='grey')
plt.xlabel('1-UniFrac',fontsize=18)
plt.ylabel('Relative Error',fontsize=18)
# plt.title('Truth vs. Relative Error')
# plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig("dmh_benchmark.pdf", dpi=300)
plt.show()
