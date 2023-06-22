import itertools
import logging
import os
import numpy as np


def generate_definitions(nr_atom_types, order):
    if order > 8:
        logging.warning(
            "Computing invariants with high order (> 8) will quickly become intractable."
        )

    # cachedir for definitions
    cachedir = "cache_generated_definitions"
    fname = f"invariants_n_atom_types_{nr_atom_types}_order_{order}.npy"
    savefile = cachedir + "/" + fname

    if os.path.exists(savefile):
        with open(savefile, "rb") as f:
            return np.load(f, allow_pickle=True)
    else:
        # generate different numbers/labels for total nr_atom_types
        labels = range(nr_atom_types)

        blocks = list(itertools.product(labels, repeat=2))

        invariants = []
        monomials_used = []

        for k in range(1, order + 1):
            monomials = itertools.product(blocks, repeat=k)

            for monomial in monomials:
                valid = True
                for i, term in enumerate(monomial):
                    if i > 0:
                        if term[0] != monomial[i - 1][1]:
                            valid = False
                            break
                # check if duplicate
                # reversed order of labels gives duplicate
                monomial_flat = [m for t in monomial for m in t]
                monomial_reversed = monomial_flat[::-1]
                valid = valid & (monomial_reversed not in monomials_used)
                # cyclic property of trace gives duplicate
                valid_tr = True
                for i, term in enumerate(monomial):
                    if i > 0:
                        monomial_shifted = monomial[i:] + monomial[:i]
                        monomial_shifted_flat = [m for t in monomial_shifted for m in t]
                        valid_tr = valid_tr & (
                            monomial_shifted_flat not in monomials_used
                        )
                        monomial_shifted_reversed = monomial_shifted_flat[::-1]
                        valid_tr = valid_tr & (
                            monomial_shifted_reversed not in monomials_used
                        )

                if valid:
                    if monomial[0][0] == monomial[-1][1]:
                        # add trace & trace complement to invariants
                        # only add trace when cyclic property of trace hasn't caused it to be a duplicate
                        # for monomials of length 1, the trace is always 0 so they're excluded
                        if (len(monomial) > 1) & valid_tr:
                            invariants.append({"type": "tr", "monomial": monomial})
                        invariants.append({"type": "tc", "monomial": monomial})
                        monomials_used.append(monomial_flat)
                    else:
                        # add sum to invariants
                        invariants.append({"type": "su", "monomial": monomial})
                        monomials_used.append(monomial_flat)

        # save invariants to cache
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        with open(savefile, "wb") as f:
            np.save(f, invariants)

        return invariants


def split_invariants_by_order(invariants):
    """
    Splits invariants by strict order.
    """
    invariants_by_order = {}

    for invariant in invariants:
        # strict order of monomial in question
        order = len(invariant["monomial"])
        if order not in invariants_by_order:
            invariants_by_order[order] = []
        invariants_by_order[order].append(invariant)

    return invariants_by_order


def count_invariants_by_order(invariants):
    """
    Counts nr of invariants by strict order (how many monomials with length 3 for example)
    """
    invariants_by_order = split_invariants_by_order(invariants)
    counts_by_order = {order: len(invs) for order, invs in invariants_by_order.items()}
    return counts_by_order


def replace_labels(invariants, label_mapping):
    """
    Replace numbers in invariants with the labels they're mapped to in label_mapping
    (in place, make copy if you don't want to alter original invariants list)

    Parameters
    ----------
    invariants: list[dict]
        list of invariants {'type': type, 'monomial': monomial} from generate_definitions
    label_mapping: dict
        dict that maps integer label to some label of choice
    """

    for invariant in invariants:
        new_monomial = []
        for pair in invariant["monomial"]:
            new_monomial.append((label_mapping[pair[0]], label_mapping[pair[1]]))
        new_monomial = tuple(new_monomial)
        invariant["monomial"] = new_monomial
