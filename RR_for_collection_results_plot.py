import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import re
import numpy as np
from collections import defaultdict

out_dir = "RR_results/validation_results/co2"

savedir = "C:/Users/amhvg/Afstudeerproject/master-thesis/images/results/pmi"

# constants
hartree_to_kcal_mol = 627.503
eV_to_kcal_mol = 23.0621

results = defaultdict(dict)

# load results
for fname in os.listdir(out_dir):
    re_abs_errors = (
        "abs_errors_(\S+)_alpha_(\S+)_solver_(\S+)_tol_(\S+)_collec_size_(\d+).npy"
    )
    re_predictions = (
        "predictions_(\S+)_alpha_(\S+)_solver_(\S+)_tol_(\S+)_collec_size_(\d+).npy"
    )
    re_labels = "labels_(\S+)_alpha_(\S+)_solver_(\S+)_tol_(\S+)_collec_size_(\d+).npy"
    if re.match(re_abs_errors, fname):
        label, alpha, solver, tol, collec_size = re.match(re_abs_errors, fname).groups()
        alpha = float(alpha)
        tol = float(tol)
        collec_size = int(collec_size)
        key = "abs_errors"
    elif re.match(re_predictions, fname):
        label, alpha, solver, tol, collec_size = re.match(
            re_predictions, fname
        ).groups()
        alpha = float(alpha)
        tol = float(tol)
        collec_size = int(collec_size)
        key = "predictions"
    elif re.match(re_labels, fname):
        label, alpha, solver, tol, collec_size = re.match(re_labels, fname).groups()
        alpha = float(alpha)
        tol = float(tol)
        collec_size = int(collec_size)
        key = "labels"

    descriptor = f"solver_{solver}_alpha_{alpha}_tol_{tol}_collec_size_{collec_size}"

    with open(out_dir + "/" + fname, "rb") as f:
        results[key][label] = np.load(f)
    
def calc_mean_abs_error(results, label, error_unit="eV"):
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

    labels = results["labels"][label] * transformation_factor
    predictions = results["predictions"][label] * transformation_factor

    return np.mean(np.abs(labels - predictions))


def cross_plot(results, label, figsize=(4, 3.9), savedir=None, error_unit="eV"):
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

    labels = results["labels"][label] * transformation_factor
    predictions = results["predictions"][label] * transformation_factor

    max_energy = max(np.max(labels), np.max(predictions))
    min_energy = min(np.min(labels), np.min(predictions))

    energy_range = max_energy - min_energy
    padding = energy_range * 0.05

    min_energy = min_energy - padding
    max_energy = max_energy + padding

    diag = np.linspace(min_energy, max_energy, 500)

    plt.figure(figsize=figsize)
    plt.title(
        f"Cross plot for best PMI collection\n({label if label == 'train' else 'validation'})"
    )
    plt.xlim(min_energy, max_energy)
    plt.ylim(min_energy, max_energy)
    plt.plot(diag, diag, color="lightgrey")
    plt.plot(labels, predictions, marker=".", linestyle="none", alpha=0.5)
    plt.xlabel(f"Real energy [{error_unit}]")
    plt.ylabel(f"Predicted energy [{error_unit}]")

    plt.tight_layout()

    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        plt.savefig(savedir + f"/cross_plot_{label}.pdf")


print("mae val eV ", calc_mean_abs_error(results, "val", error_unit="eV"))
print("mae val kcal/mol ", calc_mean_abs_error(results, "val", error_unit="kcal/mol"))
print("mae train eV ", calc_mean_abs_error(results, "train", error_unit="eV"))
print("mae train kcal/mol ", calc_mean_abs_error(results, "train", error_unit="kcal/mol"))

# cross_plot(results, "val", savedir=None)
# cross_plot(results, "train", savedir=None)

plt.show()
