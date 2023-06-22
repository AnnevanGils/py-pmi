from random_sampled_collections_sample import get_randomly_sampled_collections
from py_pmi.proximity_function import ProximityFunction1, ProximityFunction0
from py_pmi.invariant_set import InvariantSet
from py_pmi.invariant_collection import InvariantCollection
import logging
import argparse
import numpy as np

# command line parser
parser = argparse.ArgumentParser(
    description="precompute random sampled PMI  collections"
)
parser.add_argument(
    "--dataset_str_id",
    type=str,
    default="co2",
    metavar="N",
    help="dataset string identifyer, co2 or qm9",
)
parser.add_argument(
    "--collec_size", type=int, default=6, help="number of invariant sets in collection"
)
parser.add_argument(
    "--n_samples", type=int, default=20, help="number of collections sampled"
)
parser.add_argument(
    "--min_order", type=int, default=2, help="min order of a set of invariants"
)
parser.add_argument(
    "--max_order", type=int, default=5, help="max order of a set of invariants"
)
parser.add_argument(
    "--max_parallel",
    type=int,
    default=10,
    help="max number of proximity functions for which invariant values are computed at the same time",
)
parser.add_argument(
    "--compute_sequentially",
    action="store_true",
    default=False,
    help="Computation method for computing invariants in InvariantCollection",
)
parser.add_argument(
    "--savedir",
    type=str,
    default="computed_invariants",
    help="dir for saving and loading computed invariants",
)
# config
args = parser.parse_args()

seed = 21041931
np.random.seed(seed)

r0_pool = np.arange(1, 10, step=0.25)
n_pool = np.array([2, 3, 4, 5])
func_id_pool = np.array([0, 1])

# end config

# sample collections
sampled_collecs = get_randomly_sampled_collections(
    seed,
    r0_pool,
    n_pool,
    func_id_pool,
    args.collec_size,
    args.n_samples,
    args.min_order,
    args.max_order,
)


# throw everything in a collection object
# manually flatten sampled collection list into list of inv sets
inv_sets_all = []
for collec_params in sampled_collecs:
    for inv_set_params in collec_params:
        if inv_set_params["f"] == 0:
            prox_func = ProximityFunction0(inv_set_params["r0"], inv_set_params["n"])
        elif inv_set_params["f"] == 1:
            prox_func = ProximityFunction1(inv_set_params["r0"], inv_set_params["n"])
        inv_sets_all.append(
            InvariantSet(args.dataset_str_id, inv_set_params["order"], prox_func)
        )

print(f"{len(inv_sets_all)} inv sets in total")

# make one large collection object, handle duplicates internally upon calling compute
collec_all = InvariantCollection(inv_sets_all, savedir=args.savedir)
# precompute inv sets
collec_all.compute_invariants(
    max_parallel=args.max_parallel, compute_sequentially=args.compute_sequentially
)
