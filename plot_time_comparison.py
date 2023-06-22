import matplotlib.pyplot as plt
import json
import numpy as np
from collections import defaultdict
import os

fname = "results_compute_time_comparison/compute_time_comparison_results.json"

plot_savedir = "C:/Users/amhvg/Afstudeerproject/master-thesis/images/appendices/pmi"

save_plots = True

with open(fname, "r") as f:
    results = json.load(f)

order_dict_global = defaultdict(dict)

for i, (name, order_dict) in enumerate(results.items()):
    for j, (order, d) in enumerate(order_dict.items()):
        for k, key in enumerate(d.keys()):
            if i == 0 and j == 0 and k == 0:
                print(d[key].keys())
        order_dict_global[order][name] = d


def plot_time_comparison(plot_savedir, components=["total"], save_plots=False):
    for order, order_dict in order_dict_global.items():
        plt.figure()
        plt.title(f"{order}")

        for name, times_dict in order_dict.items():
            nr_prox_funcs = []
            times = {}

            for n, d in times_dict.items():
                nr_prox_funcs.append(int(n))

                for k, v in d.items():
                    if k not in times:
                        times[k] = []
                    times[k].append(np.mean(v))
                    # print(order, name, n, k, len(v))

            # print(nr_prox_funcs)
            # print(times)

            values = np.sum(np.array([np.array(times[c]) for c in components]), axis=0)
            # print(values)

            if name == "qm9_linear":
                label = "QM9 sequential"
            elif name == "qm9_padded":
                label = "QM9 (padded)"
            elif name == "co2":
                label = "CO2"
            elif name == "co2_linear":
                label = "CO2 sequential"

            plt.plot(nr_prox_funcs, values, marker=".", label=label)

        plt.ylabel("compute time [s]")
        plt.xlabel("Nr. prox. functions")
        plt.legend()
        plt.tight_layout()

        if save_plots:
            savefile = plot_savedir + "/" + f"time_comparison_{order}.pdf"

            if not os.path.exists(plot_savedir):
                os.makedirs(plot_savedir)

            # save plot
            plt.savefig(savefile)


plot_time_comparison(
    plot_savedir,
    components=[
        "loading_dataset",
        "proximity_matrices",
        "prepare_matrix_blocks",
        "compute_invariants",
    ],
    save_plots=True,
)

plt.show()
