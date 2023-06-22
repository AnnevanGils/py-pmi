from py_pmi.invariant_set import InvariantSet
from py_pmi.compute import compute_invariants, compute_invariants_variable_n_atoms
from py_pmi.generate_definitions import (
    generate_definitions,
    count_invariants_by_order,
    split_invariants_by_order,
    replace_labels,
)
from py_pmi.compute_tree import invariant_to_text
from typing import List
import logging
import os
import numpy as np
import itertools
import copy
import scipy.sparse


class InvariantCollection:
    def __init__(
        self,
        inv_sets: List[InvariantSet],
        savedir="computed_invariants",
        dtype=np.float32,
    ):
        dataset_str_id = inv_sets[0].dataset_str_id
        for inv_set in inv_sets[1:]:
            if inv_set.dataset_str_id != dataset_str_id:
                raise ValueError(
                    f"All invariant sets must have the same dataset string identifyer. Found both {dataset_str_id} and {inv_set.dataset_str_id}."
                )

        self.dataset_str_id = dataset_str_id
        self.nr_atom_types = inv_sets[0].nr_atom_types
        self.inv_sets = inv_sets
        self.dtype = dtype
        self.max_order = self._get_max_order()

        self._confirmed_no_duplicates = False

        # tuples with invariants definitions like {type: su, monomial: []}
        self.invariants_definitions = generate_definitions(
            self.nr_atom_types, self.max_order
        )

        # savedir
        self.savedir = savedir

        # load invariants by calling load_invariants
        self.invariants = None

        # load list of invariant descriptors by calling load_invariants_descriptors
        self.invariant_descriptors = None

    def _get_max_order(self):
        max_order = 1
        for inv_set in self.inv_sets:
            max_order = max(max_order, inv_set.order)
        return max_order

    def set_savedir(self, savedir: str):
        self.savedir = savedir

    def remove_duplicates(self):
        """
        removes duplicates in proximity functions, only keep version with maximum order
        """
        included_inv_sets = []
        included_inv_set_hashes = []
        included_inv_sets_orders = []
        for inv_set in self.inv_sets:
            inv_set_hash = inv_set.create_hash()

            # check if inv set hash already included, if so set order to current order if it is lower
            idx_list = np.argwhere(
                np.array(included_inv_set_hashes) == inv_set_hash
            ).flatten()
            if len(idx_list) == 1:
                idx = idx_list[0]
                # set order to current order if it is lower
                if included_inv_sets_orders[idx] < inv_set.order:
                    # replace in actual inv sets list
                    included_inv_sets[idx] = inv_set

                    # adjust in orders list
                    included_inv_sets_orders[idx] = max(
                        included_inv_sets_orders[idx], inv_set.order
                    )
            elif len(idx_list) > 1:
                raise ValueError(
                    "encountered duplicate where none should have been encountered."
                )
            else:
                # add inv set to included inv sets
                included_inv_sets.append(inv_set)
                # add inv set hash and order to included inv set hashes and orders
                included_inv_set_hashes.append(inv_set_hash)
                included_inv_sets_orders.append(inv_set.order)

        print(f"{len(included_inv_sets)} unique inv sets")

        self.inv_sets = included_inv_sets
        self._confirmed_no_duplicates = True

    def _remove_duplicates(self):
        """
        removes duplicates in proximity functions, only keep version with maximum order

        Parameters
        ----------
        inv_sets: list(InvariantSet)
            list of InvariantSet objects

        Returns
        -------
        list(InvariantSet)
            list of InvariantSet objects
        """
        included_inv_sets = []
        included_inv_set_hashes = []
        included_inv_sets_orders = []
        for inv_set in self.inv_sets:
            inv_set_hash = inv_set.create_hash()

            # check if inv set hash already included, if so set order to current order if it is lower
            idx_list = np.argwhere(
                np.array(included_inv_set_hashes) == inv_set_hash
            ).flatten()
            if len(idx_list) == 1:
                idx = idx_list[0]
                # set order to current order if it is lower
                if included_inv_sets_orders[idx] < inv_set.order:
                    # replace in actual inv sets list
                    included_inv_sets[idx] = inv_set

                    # adjust in orders list
                    included_inv_sets_orders[idx] = max(
                        included_inv_sets_orders[idx], inv_set.order
                    )
            elif len(idx_list) > 1:
                raise ValueError(
                    "encountered duplicate where none should have been encountered."
                )
            else:
                # add inv set to included inv sets
                included_inv_sets.append(inv_set)
                # add inv set hash and order to included inv set hashes and orders
                included_inv_set_hashes.append(inv_set_hash)
                included_inv_sets_orders.append(inv_set.order)

        print(f"{len(included_inv_sets)} unique inv sets")

        self._confirmed_no_duplicates = True

        return included_inv_sets

    def compute_invariants(self, max_parallel: int, compute_sequentially=False):
        """
        Computes invariant values for all inv_sets that haven't been precomputed yet.

        Parameters
        ----------
        max_parallel: int
            max number of proximity functions to compute in parallel if compute_sequentially = False
        compute_sequentually: bool
            Specify compute method to use.
            Compute using matrix products and masks for everything, use padded dataset if faster: False
            Always compute by iterating through all datapoints, variable number of atoms: True

        Returns
        -------
        None
        """
        savedir = self.savedir

        # remove duplicates if not confirmed no duplicates
        if not self._confirmed_no_duplicates:
            print("removing duplicates")
            inv_sets = self._remove_duplicates()
        else:
            inv_sets = self.inv_sets

        # see what invariants still need to be computed
        to_be_computed_by_order = {}
        for inv_set in inv_sets:
            # get dict of savefiles per order from 1 to inv_set.order
            savefiles = inv_set.get_savefiles(savedir=savedir)

            # check if highest order savefile exists, if not, invariants for it need to be computed
            max_order_savefile = savefiles[inv_set.order]
            if not os.path.exists(max_order_savefile):
                if inv_set.order not in to_be_computed_by_order:
                    to_be_computed_by_order[inv_set.order] = []
                to_be_computed_by_order[inv_set.order].append(inv_set)

        logging_statement = (
            "\n".join(
                [
                    f"\torder {order}: {len(p)} proximity functions"
                    for order, p in to_be_computed_by_order.items()
                ]
            )
            if len(to_be_computed_by_order) > 0
            else "\tNone"
        )
        print(f"Following invariants need to be computed:\n{logging_statement}")

        # compute values per order, max_parallel proximity functions at a time
        for order, inv_set_list in to_be_computed_by_order.items():
            n_remaining = len(inv_set_list)

            while n_remaining > 0:
                # take slice of size max_parallel (or less if not more points left)
                start_idx = len(inv_set_list) - n_remaining
                end_idx = min(start_idx + max_parallel, len(inv_set_list))

                inv_set_slice = inv_set_list[start_idx:end_idx]

                proximity_functions = [
                    inv_set.proximity_function for inv_set in inv_set_slice
                ]

                # adjust n_remaining accordingly
                n_remaining = len(inv_set_list) - end_idx

                # compute values
                if compute_sequentially:
                    # compute everything sequentially
                    use_sequential = True
                else:
                    # optimization structure
                    if self.dataset_str_id == "qm9":
                        if order == 2 and len(proximity_functions) > 20:
                            use_sequential = True
                        elif order == 3 and len(proximity_functions) > 12:
                            use_sequential = True
                        elif order == 4 and len(proximity_functions) > 8:
                            use_sequential = True
                        elif order == 5:
                            use_sequential = True
                        else:
                            use_sequential = False
                    else:
                        use_sequential = False

                if use_sequential:
                    values = compute_invariants_variable_n_atoms(
                        proximity_functions=proximity_functions,
                        order=order,
                        dataset_str_id=self.dataset_str_id,
                        dtype=self.dtype,
                    )
                else:
                    values = compute_invariants(
                        proximity_functions=proximity_functions,
                        order=order,
                        dataset_str_id=self.dataset_str_id,
                        dtype=self.dtype,
                    )

                print(f"computed values {values.shape}")

                # get nr of invariants per (strict) order to use to split values on (strict) order
                # counts per order for the max_order in the list of inv_sets
                counts_per_order = count_invariants_by_order(
                    self.invariants_definitions
                )

                # save values
                for i, inv_set in enumerate(inv_set_slice):
                    savefiles = inv_set.get_savefiles_create_dirs(savedir)

                    lower_idx = 0
                    for order, savefile in savefiles.items():
                        upper_idx = counts_per_order[order]
                        with open(savefile, "wb") as f:
                            np.save(
                                f,
                                values[:, i, lower_idx : lower_idx + upper_idx],
                                allow_pickle=True,
                            )
                        if order > 1:
                            lower_idx += counts_per_order[order - 1]

    def load_invariants(self, sparse=False, splits=None):
        """
        Loads invariant values for all inv_sets as one matrix, data pts on rows, invariants on columns.

        Parameters
        ----------
        sparse: bool
            return a scipy sparse coo matrix
        splits: dict
            indices for splits: {"train": list, "test": list, "val": list}

        Returns
        -------
        np matrix of (n_pts, n_invariants_total)
        """
        # print("loading invariants for collection...")

        savedir = self.savedir

        invariants_values = None
        for inv_set in self.inv_sets:
            a = None
            # print(
            #     f"loaded inv set f= {inv_set.proximity_function.function_id}, r0= {inv_set.proximity_function.r0}, n= {inv_set.proximity_function.n}, order= {inv_set.order}"
            # )
            # get dict of savefiles per order from 1 to inv_set.order
            savefiles = inv_set.get_savefiles(savedir=savedir)
            for order, savefile in savefiles.items():
                if os.path.exists(savefile):
                    with open(savefile, "rb") as f:
                        a = np.load(f) if a is None else np.hstack((a, np.load(f)))
                else:
                    raise RuntimeError(
                        f"Tried to load invariants values for inv set r0= {inv_set.proximity_function.r0}, n= {inv_set.proximity_function.n}, order= {inv_set.order}"
                    )

            invariants_values = (
                a if invariants_values is None else np.hstack((invariants_values, a))
            )

        if splits is not None:
            invariants_values = {
                label: invariants_values[indices] for label, indices in splits.items()
            }

            if sparse:
                invariants_values = {
                    label: scipy.sparse.csr_matrix(
                        (
                            invariants_split[invariants_split != 0],
                            (
                                np.argwhere(invariants_split != 0)[:, 0],
                                np.argwhere(invariants_split != 0)[:, 1],
                            ),
                        )
                    )
                    for label, invariants_split in invariants_values.items()
                }

        if sparse and splits is None:
            values = invariants_values[invariants_values != 0]
            coords = np.argwhere(invariants_values != 0)
            x_nonzero, y_nonzero = coords[:, 0], coords[:, 1]

            invariants_values = scipy.sparse.csr_matrix(
                (values, (x_nonzero, y_nonzero))
            )

        # print(f"loaded invariants for collection: {invariants_values.shape}")
        self.invariants_values = invariants_values
        return invariants_values

    def load_invariants_descriptors(self):
        """
        Loads invariant descriptor names for all inv_sets in one list, makes names unique

        Parameters
        ----------
        savedir: path to dir where invariant values and definitions should be stored and loaded from.

        Returns
        -------
        list of len n_invariants_total
        """

        def generate_invariant_descriptor(invariant, function_id, r0, n):
            return f"{invariant_to_text(invariant)}__f_{function_id}_r0_{r0}_n_{n}"

        # invariants definitions with label mapping instead of numbers
        label_mapping = self.inv_sets[0].label_to_atom_type
        invariants_definitions_explicit = copy.deepcopy(self.invariants_definitions)
        replace_labels(invariants_definitions_explicit, label_mapping)
        # explicit invariants definitions per strict order
        invariants_definitions_per_order_explicit = split_invariants_by_order(
            invariants_definitions_explicit
        )

        invs_descriptors_total = []
        for inv_set in self.inv_sets:
            r0, n, function_id = inv_set.proximity_function.get_parameters()

            invs_relevant = list(
                itertools.chain.from_iterable(
                    [
                        invariants_definitions_per_order_explicit[o + 1]
                        for o in range(inv_set.order)
                    ]
                )
            )
            invs_descriptors = [
                generate_invariant_descriptor(inv, function_id, r0, n)
                for inv in invs_relevant
            ]

            invs_descriptors_total.extend(invs_descriptors)

        self.invariant_descriptors = invs_descriptors_total
        return invs_descriptors_total
