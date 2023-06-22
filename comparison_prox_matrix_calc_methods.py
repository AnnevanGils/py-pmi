from py_pmi.proximity_function import ProximityFunction1
import numpy as np
import time

proximity_functions = [
    ProximityFunction1(r0, n) for r0 in [2.0, 3.0, 4.0] for n in [2.0, 2.0, 3.0, 3.0]
]

dtype = np.float32

# load coordinates
fname_dataset = "datasets/2CO2/2CO2_dataset_combined.npz"
with np.load(fname_dataset, allow_pickle=True) as f:
    atom_charges = f["atom_charges"]
    distances = f["distances"]

print("distances: ", distances.shape)
print("atom charges: ", atom_charges.shape)

# custom method

time_start = time.time()

n_atoms = atom_charges.shape[1]

proximity_matrices = np.ndarray(
    shape=(
        distances.shape[0],
        n_atoms,
        n_atoms,
        len(proximity_functions),
    ),
    dtype=dtype,
)
for i, prox_function in enumerate(proximity_functions):
    # proximities = prox_function.compute_proximity_matrix(distances, n_atoms)
    # print(proximities.shape)
    proximity_matrices[:, :, :, i] = prox_function.compute_proximity_matrix(
        distances, n_atoms
    )

print(proximity_matrices.shape)
# print("hgfhj")
# print(proximity_matrices[0, :, :, 0])
print(f"total time using own method: {time.time() - time_start} s")

# scipy method
time_start = time.time()

n_atoms = atom_charges.shape[1]

proximity_matrices = np.ndarray(
    shape=(
        distances.shape[0],
        n_atoms,
        n_atoms,
        len(proximity_functions),
    ),
    dtype=dtype,
)
for i, prox_function in enumerate(proximity_functions):
    # proximities = prox_function.compute_proximity_matrix(distances, n_atoms)
    # print(proximities.shape)
    proximity_matrices[:, :, :, i] = prox_function.compute_proximity_matrix_scipy(
        distances
    )

print(proximity_matrices.shape)
# print("hgfhj")
# print(proximity_matrices[0, :, :, 0])
print(f"total time using scipy squareform: {time.time() - time_start} s")
