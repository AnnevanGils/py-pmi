from py_pmi.proximity_function import ProximityFunction
from py_pmi.compute_tree import (
    get_compute_tree,
    get_categorized_compute_tree,
)
from typing import List
import numpy as np
import itertools
import time
import logging

np.set_printoptions(precision=2, linewidth=180, threshold=4000)


def compute_invariants(
    proximity_functions: List[ProximityFunction],
    order: int,
    dataset_str_id: str,
    dtype: np.dtype = np.float32,
    return_timers=False,
) -> np.ndarray:
    """
    PMIs are computed for each proximity function on the entire dataset and saved in output_dir

    Parameters
    ----------
    proximity_functions : List[ProximityFunction]
        List of objects of class ProximityFunction that PMIs are computed for
    dataset_str_id : str
        String identifyer of the dataset that the PMIs are computed for
    output_dir: str
        Path to output dir where computed PMIs should be saved
    """

    # only compute values for 10000 points when returning timers for purpose of making time comparison between datasets
    points_cap = 10000 if return_timers else None

    time_start = time.time()

    # load dataset
    if dataset_str_id.lower() == "co2":
        fname_dataset = "datasets/2CO2/2CO2_dataset_combined.npz"
        with np.load(fname_dataset) as f:
            atom_charges = f["atom_charges"][:points_cap]
            distances = f["distances"][:points_cap]
        atom_charges_for_masking = atom_charges[0]
    elif dataset_str_id.lower() == "qm9":
        fname_dataset = "datasets/QM9/QM9_dataset_padded.npz"
        with np.load(fname_dataset) as f:
            atom_charges = f["atom_charges"][:points_cap]
            distances = f["distances"][:points_cap]
            atom_charges_for_masking = f["atom_charges_for_masking"]
    else:
        raise ValueError(
            f"Dataset string identifyer '{dataset_str_id}' not known. Choose from 'CO2', 'QM9'."
        )

    # print("distances shape: ", distances.shape)
    # print("atom_charges shape: ", atom_charges.shape)
    time_intermediate = time.time()
    time_dataset_loading = time_intermediate - time_start
    logging.info(f"loaded dataset ({time_dataset_loading} s)")

    # compute proximity matrices, with
    # n_points = len(dataset),
    # n_atoms = len(number of atoms for datapoint),
    # n_prox_funcs = len(proximity functions),

    # CASE: SAME NR OF ATOMS PER DATA POINT
    # let distances be numpy matrix of shape (n_points, 1/2 * n_atoms * (n_atoms - 1)).
    # Compute proximity matrix of shape (n_points, n_prox_funcs, n_atoms, n_atoms)
    n_atoms = atom_charges.shape[1]
    n_points = distances.shape[0]
    n_proximity_functions = len(proximity_functions)

    proximity_matrices = np.ndarray(
        shape=(
            n_points,
            n_proximity_functions,
            n_atoms,
            n_atoms,
        ),
        dtype=dtype,
    )
    for i, prox_function in enumerate(proximity_functions):
        proximity_matrices[:, i, :, :] = prox_function.compute_proximity_matrix(
            distances
        )

    time_proximity_matrices = time.time() - time_intermediate
    logging.info(f"Computed proximity matrices ({time_proximity_matrices} s)")
    time_intermediate = time.time()
    logging.info(f"proximity matrices shape: {proximity_matrices.shape}")

    # define indices for block matrices using atom_charges
    # (assuming all data points have the same ordering of atoms)
    atom_charges_unique = np.unique(atom_charges_for_masking)
    label_to_atom_charge = {l: c for l, c in enumerate(atom_charges_unique)}
    label_to_bool_vector = {
        l: atom_charges == a for l, a in enumerate(atom_charges_unique)
    }
    label_to_block_size = {
        l: np.sum(atom_charges_for_masking == c)
        for l, c in label_to_atom_charge.items()
    }
    block_labels_tuples = list(itertools.product(label_to_bool_vector.keys(), repeat=2))
    block_labels_to_charges = {
        "_".join(str(l) for l in bl): (
            label_to_atom_charge[bl[0]],
            label_to_atom_charge[bl[1]],
        )
        for bl in block_labels_tuples
    }
    # print("block labels to charges: ", block_labels_to_charges)
    atom_charges_for_masking_repeated = np.repeat(
        np.expand_dims(atom_charges_for_masking, axis=0), axis=0, repeats=n_points
    )
    block_labels_to_matrix_mask = {
        bl: np.repeat(
            np.expand_dims(
                np.matmul(
                    np.expand_dims(atom_charges_for_masking_repeated == a[0], axis=-1),
                    np.expand_dims(atom_charges_for_masking_repeated == a[1], axis=-2),
                ),
                axis=1,
            ),
            repeats=n_proximity_functions,
            axis=1,
        )
        for bl, a in block_labels_to_charges.items()
    }
    # print(
    #     "block labels to matrix mask: ",
    #     {k: v.shape for k, v in block_labels_to_matrix_mask.items()},
    # )
    block_labels_to_blocks = {
        bl: proximity_matrices[m].reshape(
            n_points,
            n_proximity_functions,
            label_to_block_size[int(bl.split("_")[0])],
            label_to_block_size[int(bl.split("_")[1])],
        )
        for bl, m in block_labels_to_matrix_mask.items()
    }

    time_preparing_matrix_blocks = time.time() - time_intermediate
    logging.info(f"Prepared matrix blocks ({time_preparing_matrix_blocks} s)")
    time_intermediate = time.time()
    # print(
    #     "block labels to blocks: ",
    #     {k: v.shape for k, v in block_labels_to_blocks.items()},
    # )

    # get nested PMI dict (compute tree)
    nr_atom_types = len(atom_charges_unique)
    compute_tree, n_invariants = get_compute_tree(
        nr_atom_types=nr_atom_types, order=order
    )

    time_compute_tree = time.time() - time_intermediate
    logging.info(f"Gotten compute tree ({time_compute_tree} s)")
    logging.info(
        f"Number of invariants in total (order {order}, {nr_atom_types} atom types): {n_invariants}"
    )
    time_intermediate = time.time()

    def traverse_tree(tree, product, values):
        if "compute_info" in tree.keys():
            for invariant_type, index in tree["compute_info"].items():
                if invariant_type == "su":
                    values[:, :, index] = np.sum(product, axis=(-2, -1))
                elif invariant_type == "tr":
                    values[:, :, index] = np.trace(product, axis1=-2, axis2=-1)
                elif invariant_type == "tc":
                    values[:, :, index] = np.sum(product, axis=(-2, -1)) - np.trace(
                        product, axis1=-2, axis2=-1
                    )

        for k in tree.keys():
            if k != "compute_info":
                # get relevant block matrix
                block = block_labels_to_blocks[k]

                # calculate new product using current product
                new_product = np.matmul(product, block)

                # traverse subtree with new product
                traverse_tree(tree[k], new_product, values)

    # initialize result matrix
    # will be shaped (n_points, n_invariants, n_proximity_functions)
    values = np.zeros(
        shape=(n_points, n_proximity_functions, n_invariants), dtype=dtype
    )

    # traverse tree
    for root in compute_tree:
        # print("root: ", root)
        start_product = block_labels_to_blocks[root]
        # print(f"start product shape: {start_product.shape}")
        traverse_tree(compute_tree[root], start_product, values)

    time_computed_invariants = time.time() - time_intermediate
    logging.info(f"Computed invariants ({time_computed_invariants} s)")
    # print()
    # print("values shape: ", values.shape)
    # print("mean values: ", np.mean(values))
    time_total = time.time() - time_start
    logging.info(f"Total time: {time_total} s")

    if return_timers:
        return {
            "loading_dataset": time_dataset_loading,
            "proximity_matrices": time_proximity_matrices,
            "prepare_matrix_blocks": time_preparing_matrix_blocks,
            "compute_tree": time_compute_tree,
            "compute_invariants": time_computed_invariants,
            "total": time_total,
        }
    else:
        return values


def compute_invariants_variable_n_atoms(
    proximity_functions: List[ProximityFunction],
    order: int,
    dataset_str_id: str,
    dtype: np.dtype = np.float32,
    return_timers=False,
) -> np.ndarray:
    """
    PMIs are computed for each proximity function on the entire dataset and saved in output_dir

    Parameters
    ----------
    proximity_functions : List[ProximityFunction]
        List of objects of class ProximityFunction that PMIs are computed for
    dataset_str_id : str
        String identifyer of the dataset that the PMIs are computed for
    output_dir: str
        Path to output dir where computed PMIs should be saved
    """

    # only compute values for 10000 points when returning timers for purpose of making time comparison between datasets
    points_cap = 10000 if return_timers else None

    time_start = time.time()

    # load dataset
    if dataset_str_id.lower() == "co2":
        fname_dataset = "datasets/2CO2/2CO2_dataset_combined.npz"
        with np.load(fname_dataset) as f:
            atom_charges = f["atom_charges"][:points_cap]
            distances = f["distances"][:points_cap]
    elif dataset_str_id.lower() == "qm9":
        fname_dataset = "datasets/QM9/QM9_dataset_preprocessed.npz"
        with np.load(fname_dataset, allow_pickle=True) as f:
            atom_charges = f["atom_charges"][:points_cap]
            distances = f["distances"][:points_cap]
    else:
        raise ValueError(
            f"Dataset string identifyer '{dataset_str_id}' not known. Choose from 'CO2', 'QM9'."
        )

    # print("distances shape: ", distances.shape)
    # print("atom_charges shape: ", atom_charges.shape)
    time_intermediate = time.time()
    time_dataset_loading = time_intermediate - time_start
    logging.info(f"loaded dataset ({time_dataset_loading} s)")

    # compute proximity matrices, with
    # n_points = len(dataset),
    # n_atoms = len(number of atoms for datapoint),
    # n_prox_funcs = len(proximity functions),

    # CASE: DISTANCE MATRIX NOT SAME SHAPE PER DATA POINT
    # let distances dataset be list of len (n_points,),
    # containing for each datapoint a numpy array of shape (1/2 * n_atoms * (n_atoms - 1)).
    # For each data point, compute proximity matrices of shape (n_prox_funcs, n_atoms, n_atoms)
    n_points = atom_charges.shape[0]
    n_proximity_functions = len(proximity_functions)

    proximity_matrices = []
    atom_charges_unique_universal = set()
    for distance_array, atom_charge_array in zip(distances, atom_charges):
        n_atoms = len(atom_charge_array)

        for c in np.unique(atom_charge_array):
            atom_charges_unique_universal.add(c)

        mol_proximity_matrices = np.ndarray(
            (n_proximity_functions, n_atoms, n_atoms), dtype=dtype
        )
        for i, prox_function in enumerate(proximity_functions):
            mol_proximity_matrices[i, :, :] = prox_function.compute_proximity_matrix(
                distance_array
            )

        proximity_matrices.append(mol_proximity_matrices)

    n_atom_types_universal = len(atom_charges_unique_universal)

    time_proximity_matrices = time.time() - time_intermediate
    logging.info(f"Computed proximity matrices ({time_proximity_matrices} s)")
    time_intermediate = time.time()
    logging.info(f"proximity matrices list length: {len(proximity_matrices)}")
    logging.info(f"total number of unique atom types: {n_atom_types_universal}")

    label_to_atom_charge_universal = {
        l: c for l, c in enumerate(list(atom_charges_unique_universal))
    }
    atom_charge_to_label_universal = {
        v: k for k, v in label_to_atom_charge_universal.items()
    }

    # define indices for block matrices using atom_charges and collect unique labels per mol
    block_matrices = []
    labels_list = []
    for mol_proximity_matrices, mol_atom_charges in zip(
        proximity_matrices, atom_charges
    ):
        atom_charges_unique = np.unique(mol_atom_charges)
        label_to_bool_vector = {
            atom_charge_to_label_universal[a]: mol_atom_charges == a
            for a in atom_charges_unique
        }
        block_labels_tuples = list(
            itertools.product(label_to_bool_vector.keys(), repeat=2)
        )
        block_labels_to_charges = {
            "_".join(str(l) for l in bl): (
                label_to_atom_charge_universal[bl[0]],
                label_to_atom_charge_universal[bl[1]],
            )
            for bl in block_labels_tuples
        }
        # print("block labels to charges: ", block_labels_to_charges)
        block_labels_to_matrix_mask = {
            bl: np.repeat(
                np.expand_dims(
                    np.matmul(
                        np.expand_dims(mol_atom_charges == a[0], axis=-1),
                        np.expand_dims(mol_atom_charges == a[1], axis=-2),
                    ),
                    axis=0,
                ),
                repeats=n_proximity_functions,
                axis=0,
            )
            for bl, a in block_labels_to_charges.items()
        }
        # print(
        #     "block labels to matrix mask: ",
        #     {k: v.shape for k, v in block_labels_to_matrix_mask.items()},
        # )
        block_labels_to_blocks = {
            bl: mol_proximity_matrices[m].reshape(
                n_proximity_functions,
                np.sum(label_to_bool_vector[int(bl.split("_")[0])]),
                np.sum(label_to_bool_vector[int(bl.split("_")[1])]),
            )
            for bl, m in block_labels_to_matrix_mask.items()
        }

        block_matrices.append(block_labels_to_blocks)
        labels_list.append(
            [atom_charge_to_label_universal[a] for a in atom_charges_unique]
        )

    time_preparing_matrix_blocks = time.time() - time_intermediate
    logging.info(f"Prepared matrix blocks ({time_preparing_matrix_blocks} s)")
    time_intermediate = time.time()

    # get categorized, nested PMI dict (compute tree)
    compute_tree, n_invariants = get_categorized_compute_tree(
        nr_atom_types=n_atom_types_universal, order=order
    )

    time_compute_tree = time.time() - time_intermediate
    logging.info(f"Gotten compute tree ({time_compute_tree} s)")
    logging.info(
        f"Number of invariants in total (order {order}, {n_atom_types_universal} atom types): {n_invariants}"
    )
    time_intermediate = time.time()

    def traverse_tree(tree, product, values, mol_block_matrices):
        if "compute_info" in tree.keys():
            for invariant_type, index in tree["compute_info"].items():
                if invariant_type == "su":
                    values[:, index] = np.sum(product, axis=(-2, -1))
                elif invariant_type == "tr":
                    values[:, index] = np.trace(product, axis1=-2, axis2=-1)
                elif invariant_type == "tc":
                    values[:, index] = np.sum(product, axis=(-2, -1)) - np.trace(
                        product, axis1=-2, axis2=-1
                    )

        for k in tree.keys():
            if k != "compute_info":
                # get relevant block matrix
                block = mol_block_matrices[k]

                # calculate new product using current product
                new_product = np.matmul(product, block)

                # traverse subtree with new product
                traverse_tree(tree[k], new_product, values, mol_block_matrices)

    # initialize result matrix
    # will be shaped (n_points, n_invariants, n_proximity_functions)
    values = np.zeros(
        shape=(n_points, n_proximity_functions, n_invariants), dtype=dtype
    )

    for i, (mol_block_labels_to_blocks, mol_labels) in enumerate(
        zip(block_matrices, labels_list)
    ):
        # initialize mol values
        mol_values = np.zeros(shape=(n_proximity_functions, n_invariants), dtype=dtype)

        # construct key string from mol labels
        labels_str = "_".join([str(l) for l in sorted(mol_labels)])

        mol_compute_tree = compute_tree[labels_str]

        # traverse tree
        for root in mol_compute_tree:
            start_product = mol_block_labels_to_blocks[root]
            traverse_tree(
                mol_compute_tree[root],
                start_product,
                mol_values,
                mol_block_labels_to_blocks,
            )

        values[i] = mol_values

    time_computed_invariants = time.time() - time_intermediate
    logging.info(f"Computed invariants ({time_computed_invariants} s)")
    # print()
    # print("values shape: ", values.shape)
    # print("mean values: ", np.mean(values))
    time_total = time.time() - time_start
    logging.info(f"Total time: {time_total} s")

    if return_timers:
        return {
            "loading_dataset": time_dataset_loading,
            "proximity_matrices": time_proximity_matrices,
            "prepare_matrix_blocks": time_preparing_matrix_blocks,
            "compute_tree": time_compute_tree,
            "compute_invariants": time_computed_invariants,
            "total": time_total,
        }
    else:
        return values
