from random_sampled_collections_sample import get_randomly_sampled_collections
from py_pmi.invariant_set import InvariantSet
from py_pmi.invariant_collection import InvariantCollection
from py_pmi.proximity_function import ProximityFunction1, ProximityFunction0
import logging
import argparse
import numpy as np
import json
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.linalg import LinAlgWarning
import warnings
import scipy.sparse
import time

warnings.filterwarnings("error")

# command line parser
parser = argparse.ArgumentParser(description="RR random sampled PMI  collections")
parser.add_argument(
    "--exp_name", type=str, default="exp_1", metavar="N", help="experiment_name"
)
parser.add_argument(
    "--dataset_str_id",
    type=str,
    default="co2",
    metavar="N",
    help="dataset string identifyer, co2 or qm9",
)
parser.add_argument(
    "--sparse",
    action="store_true",
    default=False,
    help="Use sparse matrix (recommended for QM9).",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.000001,
    metavar="N",
    help="Constant that multiplies the L2 term, controlling regularization strength.",
)
parser.add_argument(
    "--solver",
    type=str,
    default="lsqr",
    metavar="N",
    help="{'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'}",
)
parser.add_argument(
    "--tol",
    type=float,
    default=0.000001,
    metavar="N",
    help="Tolerance of the solver, not used in case solver is svd or cholesky.",
)
parser.add_argument(
    "--fit_intercept",
    action="store_true",
    default=False,
    help="Whether to fit the intercept for this model. If set to false, no intercept will be used in calculations.",
)
parser.add_argument(
    "--n_train", type=int, default=30000, help="number of points in training set"
)
parser.add_argument(
    "--n_test", type=int, default=3325, help="number of points in test set"
)
parser.add_argument(
    "--n_val", type=int, default=3000, help="number of points in validation set"
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
    "--save_interval",
    type=int,
    default=10,
    help="save interval for saving errors",
)
parser.add_argument(
    "--k",
    type=int,
    default=5,
    help="k-fold cross validation",
)
parser.add_argument(
    "--savedir",
    type=str,
    default="computed_invariants",
    help="dir for saving and loading computed invariants",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="RR_results",
    help="dir for saving RR results",
)

time_start = time.time()

# config
args = parser.parse_args()

seed = 21041931
np.random.seed(seed)

r0_pool = np.arange(1, 10, step=0.25)
n_pool = np.array([2, 3, 4, 5])
func_id_pool = np.array([0, 1])

# end config

# create out dir
out_dir = f"{args.out_dir}/{args.dataset_str_id}"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if args.dataset_str_id == "co2":
    n_pts_total = 36325
elif args.dataset_str_id == "qm9":
    n_pts_total = 130831

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


def save_errors(
    errors, out_dir, label, k, alpha, solver, tol, fit_intercept, n_samples_total
):
    def get_fname_out(
        out_dir, label, k, alpha, solver, tol, fit_intercept, n_samples_total
    ):
        add = "_fit_intercept" if fit_intercept else ""
        return f"{out_dir}/mean_abs_errors_{k}-fold_{label}_alpha_{alpha}_solver_{solver}_tol_{tol}{add}_{n_samples_total}.npy"

    fname = get_fname_out(
        out_dir, label, k, alpha, solver, tol, fit_intercept, n_samples_total
    )

    with open(fname, "wb") as f:
        np.save(f, errors)


def save_metadata(
    metadata, out_dir, label, k, alpha, solver, tol, fit_intercept, n_samples_total
):
    def get_fname_out(
        out_dir, label, k, alpha, solver, tol, fit_intercept, n_samples_total
    ):
        add = "_fit_intercept" if fit_intercept else ""
        return f"{out_dir}/metadata_{k}-fold_{label}_alpha_{alpha}_solver_{solver}_tol_{tol}{add}_{n_samples_total}.npz"

    fname = get_fname_out(
        out_dir, label, k, alpha, solver, tol, fit_intercept, n_samples_total
    )

    with open(fname, "wb") as f:
        np.savez(f, **metadata)


def load_labels(dataset_str_id):
    # load dataset
    if dataset_str_id.lower() == "co2":
        fname_dataset = "datasets/2CO2/2CO2_dataset_combined.npz"
        with np.load(fname_dataset) as f:
            return f["labels_CCSD"]
    elif dataset_str_id.lower() == "qm9":
        fname_dataset = "datasets/QM9/QM9_dataset_padded.npz"
        with np.load(fname_dataset) as f:
            return f["labels_U_0"]
    else:
        raise ValueError(
            f"Dataset string identifyer '{dataset_str_id}' not known. Choose from 'CO2', 'QM9'."
        )


def make_splits(seed, n_pts_total, n_train, n_val, n_test):
    rng = np.random.default_rng(seed)
    idx_perm = rng.permutation(n_pts_total)

    # test always taken as the last .. nr points, val always as the .. nr points before that to allow different training set sizes while keeping same val & test set
    train = idx_perm[:n_train]
    test = idx_perm[-n_test:]
    val = idx_perm[-n_test - n_val : -n_test]

    return {"train": train, "val": val, "test": test}


def do_kfold(k, descriptors, labels, solver, tol, fit_intercept, alpha, seed):
    def do_RR(X_train, Y_train, X_test, Y_test, solver, tol, fit_intercept, alpha):
        reg = Ridge(
            alpha=alpha, solver=solver, tol=tol, fit_intercept=fit_intercept
        ).fit(X_train, Y_train)

        y_predict_train = reg.predict(X_train)
        signed_error_train = y_predict_train - Y_train
        abs_error_train = np.abs(signed_error_train)

        y_predict = reg.predict(X_test)
        signed_error_test = y_predict - Y_test
        abs_error_test = np.abs(signed_error_test)

        return abs_error_train, abs_error_test

    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

    try:
        abs_error_train_list = []
        abs_error_test_list = []
        for train_idx, test_idx in kfold.split(descriptors):
            X_train, Y_train = descriptors[train_idx], labels[train_idx]
            X_test, Y_test = descriptors[test_idx], labels[test_idx]

            abs_error_train, abs_error_test = do_RR(
                X_train, Y_train, X_test, Y_test, solver, tol, fit_intercept, alpha
            )

            abs_error_train_list.append(abs_error_train)
            abs_error_test_list.append(abs_error_test)

        return abs_error_train_list, abs_error_test_list
    except LinAlgWarning as w:
        print(w)
        return [np.NaN], [np.NaN]


def calc_mean_errors(error_list):
    mean_error_list = []
    for error in error_list:
        mean_error_list.append(np.mean(error))
    return mean_error_list


mean_errors_train_all = []
mean_errors_test_all = []
metadata = {"n_values": []}

# get splits indices
splits = make_splits(seed, n_pts_total, args.n_train, args.n_val, args.n_test)

for i, collec_params in enumerate(sampled_collecs):
    if i % 1 == 0:
        print(
            f"{i}/{args.n_samples} \t alpha {args.alpha} solver {args.solver}{f' tol {args.tol}' if args.solver not in ['svd', 'cholesky'] else ''}"
        )
    # prepare InvariantCollection obj
    inv_sets = []
    for inv_set_params in collec_params:
        if inv_set_params["f"] == 0:
            prox_func = ProximityFunction0(inv_set_params["r0"], inv_set_params["n"])
        elif inv_set_params["f"] == 1:
            prox_func = ProximityFunction1(inv_set_params["r0"], inv_set_params["n"])
        inv_sets.append(
            InvariantSet(args.dataset_str_id, inv_set_params["order"], prox_func)
        )
    collec = InvariantCollection(inv_sets)

    # get invariant values
    invariant_values = collec.load_invariants(args.sparse, splits)

    # get labels
    labels = load_labels(args.dataset_str_id)

    # test and val sets are swapped, train and now-test set are merged to form one big training set
    labels_train = labels[np.concatenate((splits["train"], splits["val"]))]
    labels_val = labels[splits["test"]]
    if args.sparse:
        invariants_train = scipy.sparse.vstack(
            [invariant_values["train"], invariant_values["val"]]
        )
    else:
        invariants_train = np.vstack(
            [invariant_values["train"], invariant_values["val"]]
        )
    invariants_val = invariant_values["test"]

    n_values = invariants_train.shape[1]

    # get RR errors
    abs_errors_train, abs_errors_test = do_kfold(
        args.k,
        invariants_train,
        labels_train,
        args.solver,
        args.tol,
        args.fit_intercept,
        args.alpha,
        seed,
    )

    mean_errors_train = calc_mean_errors(abs_errors_train)
    mean_errors_test = calc_mean_errors(abs_errors_test)

    mean_errors_train_all.append(mean_errors_train)
    mean_errors_test_all.append(mean_errors_test)

    metadata["n_values"].append(n_values)

    # save results intermediately
    # save mean abs errors (k errors because k fold cross validation) at save interval
    # separately for train and test errors
    if i % args.save_interval == 0:
        save_errors(
            mean_errors_train_all,
            out_dir,
            "train",
            args.k,
            args.alpha,
            args.solver,
            args.tol,
            args.fit_intercept,
            args.n_samples,
        )
        save_errors(
            mean_errors_test_all,
            out_dir,
            "test",
            args.k,
            args.alpha,
            args.solver,
            args.tol,
            args.fit_intercept,
            args.n_samples,
        )
        save_metadata(
            metadata,
            out_dir,
            "None",
            args.k,
            args.alpha,
            args.solver,
            args.tol,
            args.fit_intercept,
            args.n_samples,
        )


# save final results
save_errors(
    mean_errors_train_all,
    out_dir,
    "train",
    args.k,
    args.alpha,
    args.solver,
    args.tol,
    args.fit_intercept,
    args.n_samples,
)
save_errors(
    mean_errors_test_all,
    out_dir,
    "test",
    args.k,
    args.alpha,
    args.solver,
    args.tol,
    args.fit_intercept,
    args.n_samples,
)
save_metadata(
    metadata,
    out_dir,
    "None",
    args.k,
    args.alpha,
    args.solver,
    args.tol,
    args.fit_intercept,
    args.n_samples,
)
