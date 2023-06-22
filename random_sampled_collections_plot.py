import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import re
import numpy as np
from collections import defaultdict
from utils_connection import get_results_gum21_RR
from random_sampled_collections_sample import get_randomly_sampled_collections

cmap = mpl.colormaps["winter"]

# config
out_dir = "RR_results/gum21"
dataset_str_id = "co2"
plot_savedir = "C:/Users/amhvg/Afstudeerproject/master-thesis/images/appendices/pmi"
plot_savedir_results = (
    "C:/Users/amhvg/Afstudeerproject/master-thesis/images/results/pmi"
)
# end config

# constants
hartree_to_kcal_mol = 627.503
eV_to_kcal_mol = 23.0621

out_dir = f"{out_dir}/{dataset_str_id}"


def calc_mean_errors(errors):
    return np.mean(errors, axis=1)


def custom_dict():
    return {"data": {}}


# load results
# get_results_gum21_RR(dataset_str_id)


available_data = defaultdict(custom_dict)

# load data
for fname in os.listdir(out_dir):
    if "mean_abs_errors" in fname:
        # TODO: implement functionality for extracting fit_intercept True of False
        groups = re.match(
            "mean_abs_errors_(\d+)-fold_(\S+)_alpha_(\S+)_solver_(\w+)_tol_(?:(\S+)_(\d+)|(\S+)).npy",
            fname,
        ).groups()
        k, label, alpha, solver, tol = (
            groups[0],
            groups[1],
            groups[2],
            groups[3],
            groups[4],
        )
        config = {
            "k": int(k),
            "label": label,
            "alpha": float(alpha),
            "solver": solver,
            "tol": float(tol),
        }

        key = f"{k}-fold_alpha_{alpha}_solver_{solver}_tol_{tol}"

        with open(out_dir + "/" + fname, "rb") as f:
            a = np.load(f)

        available_data[key]["data"][label] = a
        available_data[key]["config"] = config
    elif "metadata" in fname:
        groups = re.match(
            "metadata_(\d+)-fold_(\S+)_alpha_(\S+)_solver_(\w+)_tol_(?:(\S+)_(\d+)|(\S+)).npz",
            fname,
        ).groups()
        k, label, alpha, solver, tol = (
            groups[0],
            groups[1],
            groups[2],
            groups[3],
            groups[4],
        )
        k = int(k)
        alpha = float(alpha)
        tol = float(tol)

        key = f"{k}-fold_alpha_{alpha}_solver_{solver}_tol_{tol}"

        metadata = {}
        with open(out_dir + "/" + fname, "rb") as f:
            with np.load(f, allow_pickle=True) as f2:
                for n in f2:
                    metadata[n] = f2[n]

        available_data[key]["metadata"] = metadata

# print(available_data)
for key, d in available_data.items():
    print(key)
    print(d["data"]["train"].shape)


def get_best_error_settings(
    available_data, label, collec_size=6, n_samples=1000, error_unit="eV"
):
    if error_unit == "eV":
        transformation_factor = 1
    elif error_unit == "kcal/mol":
        transformation_factor = eV_to_kcal_mol
    elif error_unit == "Ha":
        transformation_factor = eV_to_kcal_mol / hartree_to_kcal_mol
    else:
        raise ValueError(
            f"Unrecognized error_unit {error_unit}, options are eV, kcal/mol, Ha"
        )

    seed = 21041931
    r0_pool = np.arange(1, 10, step=0.25)
    n_pool = np.array([2, 3, 4, 5])
    func_id_pool = np.array([0, 1])
    min_order = 2
    max_order = 5

    sampled_collections = get_randomly_sampled_collections(
        seed,
        r0_pool,
        n_pool,
        func_id_pool,
        collec_size,
        n_samples,
        min_order,
        max_order,
    )

    lowest_mean_error = np.inf
    best_key = ""
    best_collection = None
    best_collection_idx = None

    for key, d in available_data.items():
        errors = d["data"][label]
        mean_errors = calc_mean_errors(errors)

        min_idx = np.argmin(mean_errors)

        min_error = mean_errors[min_idx]

        if min_error < lowest_mean_error:
            best_key = key
            lowest_mean_error = min_error
            best_collection = sampled_collections[min_idx]
            best_collection_idx = min_idx

    return (
        lowest_mean_error * transformation_factor,
        best_key,
        best_collection,
        best_collection_idx,
    )


def make_scatterplot(
    available_data,
    key,
    label,
    n_samples=1000,
    plot_hline=True,
    error_unit="eV",
    savedir=None,
):
    if error_unit == "eV":
        transformation_factor = 1
    elif error_unit == "kcal/mol":
        transformation_factor = eV_to_kcal_mol
    elif error_unit == "Ha":
        transformation_factor = eV_to_kcal_mol / hartree_to_kcal_mol
    else:
        raise ValueError(
            f"Unrecognized error_unit {error_unit}, options are eV, kcal/mol, Ha"
        )

    k, alpha, solver, tol = re.match(
        "(\d+)-fold_alpha_(\S+)_solver_(\S+)_tol_(\S+)", key
    ).groups()
    k = int(k)
    alpha = float(alpha)
    tol = float(tol)

    title = f"Ridge regression {k}-fold mean {label} errors for {n_samples} sampled PMI combinations                       "
    subtitle = (
        rf"solver: {solver}, $\alpha$: {alpha}"
        if solver in ["svd", "cholesky"]
        else rf"solver: {solver}, $\alpha$: {alpha}, tol.: {tol}"
    )

    # hline height (1 kcal/mol) in eV multiplied by transformation factor
    hline_height = 1 / eV_to_kcal_mol * transformation_factor

    # make plot
    d = available_data[key]
    metadata = d["metadata"]["n_values"]
    errors = d["data"][label]

    mean_errors = calc_mean_errors(errors)
    x_pos = np.random.rand(mean_errors.shape[0])

    min_idx = np.argmin(mean_errors)
    min_error = mean_errors[min_idx]
    min_pos = x_pos[min_idx]

    plt.figure(figsize=(10, 4))
    plt.suptitle(title)
    plt.title(subtitle)
    # plot horizontal line at 1 kcal/mol
    if plot_hline:
        plt.axhline(hline_height, c="lightgrey")

    plt.scatter(
        x_pos,
        mean_errors * transformation_factor,
        c=metadata,
        cmap=cmap,
    )
    perc_5_line = np.percentile(mean_errors * transformation_factor, 5)
    plt.axhline(perc_5_line, c="red", label="5-percentile")
    plt.plot(
        min_pos,
        min_error * transformation_factor,
        c="red",
        marker="o",
        linestyle="none",
        fillstyle="none",
        label="best result",
    )
    plt.xlabel("random position")
    plt.xticks([])
    plt.ylabel(f"mean abs error [{error_unit}]")
    plt.legend()
    cbar = plt.colorbar()
    cbar.set_label("Total nr. of invariants in collection", labelpad=15, rotation=270)

    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        fname = savedir + "/" + f"scatterplot_best_settings_best_{label}.pdf"

        plt.savefig(fname)


def plot_histograms(available_data, error_unit="eV"):
    if error_unit == "eV":
        transformation_factor = 1
    elif error_unit == "kcal/mol":
        transformation_factor = eV_to_kcal_mol
    elif error_unit == "Ha":
        transformation_factor = eV_to_kcal_mol / hartree_to_kcal_mol
    else:
        raise ValueError(
            f"Unrecognized error_unit {error_unit}, options are eV, kcal/mol, Ha"
        )

    # TODO: colormap for number of values in feature vector

    # make plots
    for key, d in available_data.items():
        metadata = d["metadata"]["n_values"]
        # print()
        # print(metadata)
        data = d["data"]
        for label, errors in data.items():
            plt.figure()
            plt.title(f"{key} {label}")
            mean_errors = calc_mean_errors(errors)
            print(key, mean_errors.shape[0])
            plt.hist(mean_errors * transformation_factor, rwidth=0.9)
            plt.xlabel(f"mean abs error [{error_unit}]")

            # get 90-percentile
            print(
                f"90 percentile [{error_unit}]: ",
                np.percentile(mean_errors * transformation_factor, 90),
            )
            print(
                f"5 percentile [{error_unit}]: ",
                np.percentile(mean_errors * transformation_factor, 5),
            )


def plot_percentiles(
    available_data,
    q,
    label="test",
    plot_hline=True,
    figsize=(6, 4),
    error_unit="eV",
    savedir=None,
):
    if error_unit == "eV":
        transformation_factor = 1
    elif error_unit == "kcal/mol":
        transformation_factor = eV_to_kcal_mol
    elif error_unit == "Ha":
        transformation_factor = eV_to_kcal_mol / hartree_to_kcal_mol
    else:
        raise ValueError(
            f"Unrecognized error_unit {error_unit}, options are eV, kcal/mol, Ha"
        )

    # hline height (1 kcal/mol) in eV multiplied by transformation factor
    hline_height = 1 / eV_to_kcal_mol * transformation_factor

    # collect plot data per solver, 5-percentile error as function of alpha
    plot_data = defaultdict(lambda: {"alpha": [], "error": []})
    for key, d in available_data.items():
        k, alpha, solver, tol = re.match(
            "(\d+)-fold_alpha_(\S+)_solver_(\S+)_tol_(\S+)", key
        ).groups()
        k = int(k)
        alpha = float(alpha)
        tol = float(tol)

        errors = d["data"][label]
        mean_errors = calc_mean_errors(errors)
        perc_q_value = np.percentile(mean_errors * transformation_factor, q)

        plot_data[f"{solver}_{tol}"]["alpha"].append(alpha)
        plot_data[f"{solver}_{tol}"]["error"].append(perc_q_value)

    plt.figure(figsize=figsize)
    plt.title(f"{q}-percentile error on {label}")
    plt.xscale("log")

    # plot horizontal line at 1 kcal/mol
    if plot_hline:
        plt.axhline(hline_height, c="lightgrey")

    for solver, data in plot_data.items():
        solv, tol = solver.split("_")
        tol = float(tol)
        alpha = np.array(data["alpha"])
        error = np.array(data["error"])
        idx = np.argsort(alpha)
        alpha = alpha[idx]
        error = error[idx]
        plt.plot(
            alpha,
            error,
            marker="o",
            label=f"{solv}{f' (tol. {tol:.0e})' if solv not in ['svd', 'cholesky'] else ''}",
        )

        idx_min_error = np.argmin(error)

        print(
            f"alpha with lowest {q}-percentile error for solver {solver}: ",
            alpha[idx_min_error],
        )

    plt.xlabel(r"$\alpha$")
    plt.ylabel(f"mean abs error [{error_unit}]")
    plt.legend()

    plt.tight_layout()

    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        fname = savedir + "/" + f"solvers_over_alpha_{q}-percentile_{label}.pdf"

        plt.savefig(fname)


def plot_best(
    available_data, label="test", plot_hline=True, error_unit="eV", savedir=None
):
    if error_unit == "eV":
        transformation_factor = 1
    elif error_unit == "kcal/mol":
        transformation_factor = eV_to_kcal_mol
    elif error_unit == "Ha":
        transformation_factor = eV_to_kcal_mol / hartree_to_kcal_mol
    else:
        raise ValueError(
            f"Unrecognized error_unit {error_unit}, options are eV, kcal/mol, Ha"
        )

    # hline height (1 kcal/mol) in eV multiplied by transformation factor
    hline_height = 1 / eV_to_kcal_mol * transformation_factor

    # collect plot data per solver, 5-percentile error as function of alpha
    plot_data = defaultdict(lambda: {"alpha": [], "error": []})
    for key, d in available_data.items():
        k, alpha, solver, tol = re.match(
            "(\d+)-fold_alpha_(\S+)_solver_(\S+)_tol_(\S+)", key
        ).groups()
        k = int(k)
        alpha = float(alpha)
        tol = float(tol)

        errors = d["data"][label]
        mean_errors = calc_mean_errors(errors)
        min_value = np.min(mean_errors * transformation_factor)

        plot_data[f"{solver}_{tol}"]["alpha"].append(alpha)
        plot_data[f"{solver}_{tol}"]["error"].append(min_value)

    plt.figure()
    plt.title(f"best error on {label}")
    plt.xscale("log")

    # plot horizontal line at 1 kcal/mol
    if plot_hline:
        plt.axhline(hline_height, c="lightgrey")

    for solver, data in plot_data.items():
        solv, tol = solver.split("_")
        alpha = np.array(data["alpha"])
        error = np.array(data["error"])
        idx = np.argsort(alpha)
        alpha = alpha[idx]
        error = error[idx]
        plt.plot(alpha, error, marker="o", label=f"{solv} (tol. {tol})")

        idx_min_error = np.argmin(error)

        print(
            f"alpha with lowest best error for solver {solver}: ", alpha[idx_min_error]
        )

    plt.xlabel(r"$\alpha$")
    plt.ylabel(f"mean abs error [{error_unit}]")
    plt.legend()

    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        fname = savedir + "/" + f"solvers_over_alpha_best_{label}.pdf"

        plt.savefig(fname)


(
    lowest_mean_error,
    best_key,
    best_collection,
    best_collection_idx,
) = get_best_error_settings(available_data, label="test", error_unit="eV")

print(best_collection)
print(lowest_mean_error)

# make_scatterplot(
#     available_data,
#     best_key,
#     label="test",
#     error_unit="kcal/mol",
#     savedir=plot_savedir_results,
# )
# plot_percentiles(available_data, 5, savedir=plot_savedir)
# plot_percentiles(available_data, 5, label="train")
# plot_best(available_data)
# plot_best(available_data, label="train")

plt.show()
