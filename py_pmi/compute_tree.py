from py_pmi.generate_definitions import generate_definitions, replace_labels
import itertools
import copy
import logging
import time


def invariant_to_text(invariant):
    s = invariant["type"]
    for pair in invariant["monomial"]:
        s += f"__{pair[0]}_{pair[1]}"
    return s


def create_compute_tree(invariants, index_mapping):
    """
    Parameters
    ----------
    invariants: list[dict]
        list of invariants {'type': type, 'monomial': monomial} from generate_definitions
    index_mapping: dict
        dict that maps integer label to global index that indicates the position in the order
        in which all the invariants are generated

    Returns
    -------
    dict
        Compute tree with stringified monomials as keys, containing a subtree for following terms
        in remainder of monomial and compute information {type: index}
    """

    def add_to_tree(invariant, remaining_monomial, tree):
        # monomial_txt = "__".join([f"{m[0]}_{m[1]}" for m in invariant["monomial"]])
        if len(remaining_monomial) == 0:
            if "compute_info" in tree:
                tree["compute_info"][invariant["type"]] = index_mapping[
                    invariant_to_text(invariant)
                ]
            else:
                tree["compute_info"] = {
                    invariant["type"]: index_mapping[invariant_to_text(invariant)]
                }
        else:
            m = remaining_monomial[0]
            m_txt = f"{m[0]}_{m[1]}"
            if m_txt not in tree:
                tree[m_txt] = {}
            add_to_tree(invariant, remaining_monomial[1:], tree[m_txt])

    # generate compute tree
    compute_tree = {}
    for invariant in invariants:
        add_to_tree(invariant, invariant["monomial"], compute_tree)

    return compute_tree


def get_compute_tree(nr_atom_types, order):
    """
    Parameters
    ----------
    nr_atom_types: int
        number of distinct atom types in the system
    order: int
        order of the monomials in the invariants
    Returns
    -------
    dict
        Compute tree with stringified monomials as keys, containing a subtree for following terms
        in remainder of monomial and compute information {type: index}
    int
        number of invariants in total for this nr atom types and max order
    """

    # get invariants definitions (all)
    invariants = generate_definitions(nr_atom_types, order)
    n_invariants = len(invariants)

    # map invariants to index for the order in which they're returned by generate_definitions
    invariant_to_index = {
        invariant_to_text(invariant): i for i, invariant in enumerate(invariants)
    }

    compute_tree = create_compute_tree(invariants, invariant_to_index)

    return compute_tree, n_invariants


def get_categorized_compute_tree(nr_atom_types, order):
    """
    Parameters
    ----------
    nr_atom_types: int
        number of distinct atom types in the system
    order: int
        order of the monomials in the invariants
    Returns
    -------
    dict
        Dict with as keys all possible combinations of atom_types of length 1 up to nr_atom_types,
        each key containing a compute tree for invariants for atom_types relevant for that key
    int
        number of invariants in total for this nr atom types and max order
    """

    # get invariants definitions (all)
    invariants = generate_definitions(nr_atom_types, order)
    n_invariants = len(invariants)

    # map invariants to index for the order in which they're returned by generate_definitions
    invariant_to_index = {
        invariant_to_text(invariant): i for i, invariant in enumerate(invariants)
    }

    # create combinations
    combinations = list(
        itertools.chain.from_iterable(
            itertools.combinations(range(nr_atom_types), r=length)
            for length in range(1, nr_atom_types + 1)
        )
    )

    # get invariants definitions per combinations length
    invariants_per_length = {
        length: generate_definitions(nr_atom_types=length, order=order)
        for length in range(1, nr_atom_types)
    }
    invariants_per_length[nr_atom_types] = invariants

    categorized_compute_tree = {}
    for c in combinations:
        # text indicator for combination
        c_txt = "_".join([str(i) for i in c])

        # take invariants for the correct nr of atom labels in combination
        c_invariants = copy.deepcopy(invariants_per_length[len(c)])

        # replace labels in c_invariants with those relevant for this combination
        label_mapping = {j: i for j, i in enumerate(c)}
        replace_labels(c_invariants, label_mapping=label_mapping)

        # add compute tree with mapped labels and global indices to categorized compute tree
        categorized_compute_tree[c_txt] = create_compute_tree(
            c_invariants, index_mapping=invariant_to_index
        )

    return categorized_compute_tree, n_invariants
