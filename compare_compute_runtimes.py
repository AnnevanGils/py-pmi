from py_pmi.compute import compute_invariants, compute_invariants_variable_n_atoms
from py_pmi.proximity_function import ProximityFunction1
import logging
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

orders = [2, 3, 4, 5][3:]
n_prox_functions = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24]
n_runs_for_avg = 5

prox_functions = [
    ProximityFunction1(r0, n)
    for r0 in [1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 5.0]
    for n in [2.0, 3.0, 4.0]
]

# overwrite output file
# with open("compute_time_comparison_results.json", "w") as f:
#     json.dump({}, f)


def add_to_time_results(name, order, n_prox_funcs, times):
    fname = "compute_time_comparison_results.json"

    if os.path.exists(fname):
        with open(fname, "r") as f:
            time_results = json.load(f)
    else:
        time_results = {}

    order_key = f"order_{order}"
    n_prox_funcs = str(n_prox_funcs)
    if name not in time_results:
        time_results[name] = {}
    if order_key not in time_results[name]:
        time_results[name][order_key] = {}
    if n_prox_funcs not in time_results[name][order_key]:
        time_results[name][order_key][n_prox_funcs] = {}
    for k, v in times.items():
        if k not in time_results[name][order_key][n_prox_funcs]:
            time_results[name][order_key][n_prox_funcs][k] = []
        time_results[name][order_key][n_prox_funcs][k].append(v)

    with open(fname, "w") as f:
        json.dump(time_results, f)


for order in orders:
    for stop_idx in n_prox_functions:
        if order == 4:
            if stop_idx <= 20:
                continue
            else:
                n_runs_for_avg_temp = n_runs_for_avg
        else:
            n_runs_for_avg_temp = n_runs_for_avg

        print(f"order {order}, {stop_idx} prox functions")
        for run in range(n_runs_for_avg_temp):
            # qm9 padded
            print(f"starting qm9 padded (run {run})")
            times = compute_invariants(
                prox_functions[:stop_idx],
                order=order,
                dataset_str_id="qm9",
                return_timers=True,
            )

            add_to_time_results("qm9_padded", order, stop_idx, times)

            # qm9 linear
            print(f"starting qm9 linear (run {run})")
            times = compute_invariants_variable_n_atoms(
                prox_functions[:stop_idx],
                order=order,
                dataset_str_id="qm9",
                return_timers=True,
            )

            add_to_time_results("qm9_linear", order, stop_idx, times)

            # co2
            print(f"starting co2 (run {run})")
            times = compute_invariants(
                prox_functions[:stop_idx],
                order=order,
                dataset_str_id="co2",
                return_timers=True,
            )

            add_to_time_results("co2", order, stop_idx, times)

            # co2 linear
            print(f"starting co2 linear (run {run})")
            times = compute_invariants_variable_n_atoms(
                prox_functions[:stop_idx],
                order=order,
                dataset_str_id="co2",
                return_timers=True,
            )

            add_to_time_results("co2_linear", order, stop_idx, times)

        print()
