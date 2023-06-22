from py_pmi.generate_definitions import generate_definitions
import json
from collections import defaultdict

orders = [1, 2, 3, 4, 5]
n_atom_types = [1, 2, 3, 4, 5][1:2]

print(f"order: {orders}, n_atom_types: {n_atom_types}")

outf = "invariant_counts_overview.json"

results = defaultdict(dict)

for order in orders:
    for n_atoms in n_atom_types:
        invariants = generate_definitions(n_atoms, order)
        print(order, n_atoms, len(invariants))
        results[order][n_atoms] = len(invariants)

        # with open(outf, "w") as f:
        #     json.dump(results, f)

print(results)

# with open(outf, "w") as f:
#     json.dump(results, f)
