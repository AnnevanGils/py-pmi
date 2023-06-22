from py_pmi.invariant_set import InvariantSet
from py_pmi.invariant_collection import InvariantCollection
from py_pmi.proximity_function import ProximityFunction1, ProximityFunction0
import argparse
import numpy as np
import os
from sklearn.linear_model import Ridge

# warnings.filterwarnings("error")

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

# config
args = parser.parse_args()

seed = 21041931
np.random.seed(seed)

if args.dataset_str_id == "co2":
    # best collection from parameter scan (solver svd, alpha 1e-6)
    collec_params = [
        {"f": 1, "r0": 9.75, "n": 2, "order": 5},
        {"f": 1, "r0": 8.75, "n": 2, "order": 5},
        {"f": 0, "r0": 7.25, "n": 3, "order": 5},
        {"f": 1, "r0": 3.75, "n": 3, "order": 4},
        {"f": 0, "r0": 8.5, "n": 3, "order": 5},
        {"f": 0, "r0": 4.5, "n": 4, "order": 5},
    ]
    collec_size = len(collec_params)
elif args.dataset_str_id == "qm9":
    pass

if args.dataset_str_id == "co2":
    n_pts_total = 36325
elif args.dataset_str_id == "qm9":
    n_pts_total = 130831


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


def do_RR(X_train, Y_train, X_test, Y_test, solver, tol, fit_intercept, alpha):
    reg = Ridge(alpha=alpha, solver=solver, tol=tol, fit_intercept=fit_intercept).fit(
        X_train, Y_train
    )

    y_predict_train = reg.predict(X_train)
    signed_error_train = y_predict_train - Y_train
    abs_error_train = np.abs(signed_error_train)

    y_predict = reg.predict(X_test)
    signed_error_test = y_predict - Y_test
    abs_error_test = np.abs(signed_error_test)

    return abs_error_train, abs_error_test, y_predict_train, y_predict


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

# compute invariants
# collec.compute_invariants(max_parallel=10)

# get invariant values
invariant_values = collec.load_invariants()
n_values = invariant_values.shape[1]

# get labels
labels = load_labels(args.dataset_str_id)

# get splits indices
splits = make_splits(seed, n_pts_total, args.n_train, args.n_val, args.n_test)

labels_train = labels[np.concatenate((splits["train"], splits["val"]))]
labels_val = labels[splits["test"]]
invariants_train = invariant_values[np.concatenate((splits["train"], splits["val"]))]
invariants_val = invariant_values[splits["test"]]

# do ridge regression, get abs error values for train set and validation set
abs_error_train, abs_error_val, predictions_train, predictions_val = do_RR(
    invariants_train,
    labels_train,
    invariants_val,
    labels_val,
    solver=args.solver,
    tol=args.tol,
    fit_intercept=args.fit_intercept,
    alpha=args.alpha,
)

print("mean abs error train (eV): ", np.mean(abs_error_train))
print("mean abs error val (eV): ", np.mean(abs_error_val))

# save results
# create out dir
out_dir = f"{args.out_dir}/validation_results/{args.dataset_str_id}"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def get_fname(label, out_dir, alpha, solver, tol, fit_intercept, collec_size):
    add = "_fit_intercept" if fit_intercept else ""
    return f"{out_dir}/abs_errors_{label}_alpha_{alpha}_solver_{solver}_tol_{tol}{add}_collec_size_{collec_size}.npy"


def get_fname_predictions(
    label, out_dir, alpha, solver, tol, fit_intercept, collec_size
):
    add = "_fit_intercept" if fit_intercept else ""
    return f"{out_dir}/predictions_{label}_alpha_{alpha}_solver_{solver}_tol_{tol}{add}_collec_size_{collec_size}.npy"


def get_fname_labels(label, out_dir, alpha, solver, tol, fit_intercept, collec_size):
    add = "_fit_intercept" if fit_intercept else ""
    return f"{out_dir}/labels_{label}_alpha_{alpha}_solver_{solver}_tol_{tol}{add}_collec_size_{collec_size}.npy"


# save train
fname_train = get_fname(
    "train", out_dir, args.alpha, args.solver, args.tol, args.fit_intercept, collec_size
)
with open(fname_train, "wb") as f:
    np.save(f, abs_error_train)

fname_predictions_train = get_fname_predictions(
    "train", out_dir, args.alpha, args.solver, args.tol, args.fit_intercept, collec_size
)
with open(fname_predictions_train, "wb") as f:
    np.save(f, predictions_train)

fname_labels_train = get_fname_labels(
    "train", out_dir, args.alpha, args.solver, args.tol, args.fit_intercept, collec_size
)
with open(fname_labels_train, "wb") as f:
    np.save(f, labels_train)

# save val
fname_val = get_fname(
    "val", out_dir, args.alpha, args.solver, args.tol, args.fit_intercept, collec_size
)
with open(fname_val, "wb") as f:
    np.save(f, abs_error_val)

fname_predictions_val = get_fname_predictions(
    "val", out_dir, args.alpha, args.solver, args.tol, args.fit_intercept, collec_size
)
with open(fname_predictions_val, "wb") as f:
    np.save(f, predictions_val)

fname_labels_val = get_fname_labels(
    "val", out_dir, args.alpha, args.solver, args.tol, args.fit_intercept, collec_size
)
with open(fname_labels_val, "wb") as f:
    np.save(f, labels_val)
