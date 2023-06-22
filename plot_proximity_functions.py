import matplotlib.pyplot as plt
from py_pmi.proximity_function import ProximityFunction0, ProximityFunction1
import json
import numpy as np
import os

savedir = "C:/Users/amhvg/Afstudeerproject/master-thesis/images/results/pmi"

with open("invariant_counts_overview.json", "r") as f:
    counts = json.load(f)

print(counts)

collec_params = [
    {"f": 1, "r0": 9.75, "n": 2, "order": 5},
    {"f": 1, "r0": 8.75, "n": 2, "order": 5},
    {"f": 0, "r0": 7.25, "n": 3, "order": 5},
    {"f": 1, "r0": 3.75, "n": 3, "order": 4},
    {"f": 0, "r0": 8.5, "n": 3, "order": 5},
    {"f": 0, "r0": 4.5, "n": 4, "order": 5},
]

proximity_functions = []
ids = []

for inv_set in collec_params:
    order = inv_set["order"]
    print(inv_set, counts[str(order)]["2"])

    ids.append(inv_set["f"])

    if inv_set["f"] == 0:
        proximity_functions.append(ProximityFunction0(r0=inv_set["r0"], n=inv_set["n"]))
    elif inv_set["f"] == 1:
        proximity_functions.append(ProximityFunction1(r0=inv_set["r0"], n=inv_set["n"]))

ids = np.array(ids)
sorted_idx = np.argsort(ids)
proximity_functions = np.array(proximity_functions)
proximity_functions = proximity_functions[sorted_idx]

plt.figure(figsize=(6, 4))
plt.title("Proximity functions in best PMI combination")

r_range = np.linspace(0, 10, 1000)

for prox_func in proximity_functions:
    r0, n, f = prox_func.get_parameters()
    values = prox_func.proximity_function(r_range)

    plt.plot(r_range, values, label=rf"$f_{f} \leftarrow r_0$: {r0}, $n$: {n}")

plt.xlabel("r [Å]")
plt.ylabel("proximity scaling factor")
plt.legend()

plt.tight_layout()

if not os.path.exists(savedir):
    os.makedirs(savedir)

fname = savedir + "/" + f"proximity_functions_best_collection.pdf"

plt.savefig(fname)

plt.show()
