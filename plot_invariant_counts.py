import matplotlib.pyplot as plt
import json
from collections import defaultdict
import os

fname = "invariant_counts_overview.json"

plot_savedir = "C:/Users/amhvg/Afstudeerproject/master-thesis/images/appendices/pmi"

save_plot = True

with open(fname, "r") as f:
    results = json.load(f)


plot_data = defaultdict(lambda: {"counts": [], "orders": []})

for order, n_atoms_dict in results.items():
    for n_atoms, count in n_atoms_dict.items():
        plot_data[n_atoms]["counts"].append(count)
        plot_data[n_atoms]["orders"].append(order)

plt.figure()
plt.yscale("log")
plt.title("Number of distinct invariants per order")

markers = [".", "x", "*", "^", "s"]

for n_atoms, d in plot_data.items():
    plt.plot(
        d["orders"],
        d["counts"],
        marker=markers[int(n_atoms) - 1],
        label=f"{n_atoms} atom types",
    )

plt.xlabel("order")
plt.ylabel("Nr. invariants")
plt.legend()

plt.tight_layout()

if save_plot:
    outf = plot_savedir + "/" + "invariant_counts_overview.pdf"

    if not os.path.exists(plot_savedir):
        os.makedirs(plot_savedir)

    plt.savefig(outf)

plt.show()
