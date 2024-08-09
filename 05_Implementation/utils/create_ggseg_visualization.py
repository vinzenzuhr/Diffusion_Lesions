import sys

import ggseg 
import matplotlib.pyplot as plt

file = sys.argv[1]
output_dir = sys.argv[2]

data = dict()
with open(file, "r") as f:
    for line in f:
        key, value = line.split(",")
        if key[:3] == "lh.":
            key = key[3:] + "_left"
        elif key[:3] == "rh.":
            key = key[3:] + "_right"
        else:  
            raise ValueError("Key has to start with 'lh.' or 'rh.'")
        data[key] = float(value[:-2])

ggseg.plot_dk(data, ylabel="Absolute changes relative to the mean (%)", cmap="bwr", vminmax=[0, 3.0])
plt.savefig(output_dir + "/ggseg_plot.png")

