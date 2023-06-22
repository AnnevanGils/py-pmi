import numpy as np
import math


def calc_nr_combinations(total_nr_inv_sets_options, collec_size):
    return math.factorial(total_nr_inv_sets_options) / (
        math.factorial(collec_size)
        * math.factorial(total_nr_inv_sets_options - collec_size)
    )


def get_randomly_sampled_collections(
    seed, r0_pool, n_pool, func_id_pool, collec_size, n_samples, min_order, max_order
):
    """
    sample collecs randomly, proximity function parameters sampled from pool according to uniform random distr

    Parameters
    ----------
    seed
        seed for numpy random
    r0_pool: np.array(float)
        r0 values pool for proximity function
    n_pool: np.array(float)
        n values pool for proximity function
    func_id_pool: np.array(int)
        function identifyer pool
    collec_size: int
        number of InvariantSets in an InvariantCollection
    n_samples: int
        number of collections to be sampled
    min_order: int
        min order of a set of invariants
    max_order: int
        max_order of a set of invariants

    Returns
    -------
    list
        list of numpy arrays with collec_size number of dicts inside
        where the dict has keys f, r0, n, order representing an InvariantSet
    """
    np.random.seed(seed)

    def create_collec_hash(collec_params):
        inv_set_params_list = [
            f"f_{d['f']}_r0_{d['r0']}_n_{d['n']}" for d in collec_params
        ]
        inv_set_params_list.sort()
        return "__".join(inv_set_params_list)

    def sample_unique_collec(sampled_collec_hashes, pool, collec_size):
        while True:
            # sample new collec as list of collec_size inv_set_params from pool without replacement
            inv_set_params_idx = np.random.choice(len(pool), collec_size, replace=False)
            new_collec = pool[inv_set_params_idx]

            # check if random sampled collec not in sampled_collecs
            new_collec_hash = create_collec_hash(new_collec)
            if new_collec_hash not in sampled_collec_hashes:
                sampled_collec_hashes.append(new_collec_hash)
                return new_collec
            else:
                print("unicorn")

    # fill pool of inv set params options
    inv_set_params_pool = []
    for f in func_id_pool:
        for r0 in r0_pool:
            for n in n_pool:
                inv_set_params_pool.append({"f": f, "r0": r0, "n": n})
    inv_set_params_pool = np.array(inv_set_params_pool)

    # some stats
    n_inv_set_options = len(r0_pool) * len(n_pool) * len(func_id_pool)
    print(f"nr of inv set options: {n_inv_set_options}")
    print(f"actual inv set params pool len: {len(inv_set_params_pool)}")
    print(
        f"nr of collec options: {calc_nr_combinations(n_inv_set_options, collec_size)}"
    )

    # sample random collections (list of inv set params)
    sampled_collecs = []
    sampled_collec_hashes = []
    while len(sampled_collecs) < n_samples:
        sampled_collecs.append(
            sample_unique_collec(
                sampled_collec_hashes, inv_set_params_pool, collec_size
            )
        )

    # add random order to sampled collections' inv set params
    for inv_set_params_list in sampled_collecs:
        for inv_set_params in inv_set_params_list:
            inv_set_params["order"] = np.random.randint(min_order, max_order + 1)

    return sampled_collecs
