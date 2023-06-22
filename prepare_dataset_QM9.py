import numpy as np
import os
import csv
import re
from scipy.spatial.distance import pdist, squareform
from sortedcontainers import SortedDict
import itertools

np.set_printoptions(precision=2, linewidth=3000, threshold=2000)

datadir = "../QM9/dsgdb9nsd.xyz"
file_exclude_mols = "../QM9/uncharacterized.csv"

# get ids to exclude
excluded_ids = []
with open(file_exclude_mols, newline="") as f:
    reader = csv.reader(f, delimiter=";")
    for row in reader:
        excluded_ids.append(int(row[1]))

# initialize data lists
nr_atoms = []
mol_ids = []
atom_charges = []
coordinates = []
distances = []
labels_rot_A = []
labels_rot_B = []
labels_rot_C = []
labels_mu = []
labels_alpha = []
labels_eps_homo = []
labels_eps_lumo = []
labels_eps_gap = []
labels_elec_spatial_extent = []
labels_zpve = []
labels_U_0 = []
labels_U = []
labels_H = []
labels_G = []
labels_C_v = []

# atom types to charges
atom_type_to_charge = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

total_files = os.listdir(datadir)

for i, fname in enumerate(total_files):
    if re.match("dsgdb9nsd_(\d+).xyz", fname):
        mol_id = int(re.match("dsgdb9nsd_(\d+).xyz", fname).groups()[0])

        if i % 1000 == 0:
            print(i, "/", len(total_files))

        if mol_id not in excluded_ids:
            mol_ids.append(mol_id)

            # read xyz file
            with open(datadir + "/" + fname, "r") as f:
                nr_atoms_file = int(f.readline())
                nr_atoms.append(nr_atoms_file)
                (
                    tag_gdb_id,
                    rot_A,
                    rot_B,
                    rot_C,
                    mu,
                    alpha,
                    eps_homo,
                    eps_lumo,
                    eps_gap,
                    elec_spatial_extent,
                    zpve,
                    U_0,
                    U,
                    H,
                    G,
                    C_v,
                ) = [
                    np.float64(n) if j > 0 else n
                    for j, n in enumerate(f.readline().strip("\n").split("\t")[:-1])
                ]

                labels_rot_A.append(rot_A)
                labels_rot_B.append(rot_B)
                labels_rot_C.append(rot_C)
                labels_mu.append(mu)
                labels_alpha.append(alpha)
                labels_eps_homo.append(eps_homo)
                labels_eps_lumo.append(eps_lumo)
                labels_eps_gap.append(eps_gap)
                labels_elec_spatial_extent.append(elec_spatial_extent)
                labels_zpve.append(zpve)
                labels_U_0.append(U_0)
                labels_U.append(U)
                labels_H.append(H)
                labels_G.append(G)
                labels_C_v.append(C_v)

                atom_charges_mol = np.zeros(nr_atoms_file)
                coordinates_mol = np.zeros((nr_atoms_file, 3))
                for n in range(nr_atoms_file):
                    atom_type, x, y, z, mul_part_charge = [
                        np.float64(p) if j > 0 else p
                        for j, p in enumerate(f.readline().strip("\n").split("\t"))
                    ]
                    atom_charges_mol[n] = atom_type_to_charge[atom_type]
                    coordinates_mol[n, :] = np.array([x, y, z])

                atom_charges.append(atom_charges_mol)
                coordinates.append(coordinates_mol)

                distances.append(pdist(coordinates_mol))

data_out = {
    "nr_atoms": nr_atoms,
    "mol_ids": mol_ids,
    "atom_charges": atom_charges,
    "coordinates": coordinates,
    "distances": distances,
    "labels_rot_A": labels_rot_A,
    "labels_rot_B": labels_rot_B,
    "labels_rot_C": labels_rot_C,
    "labels_mu": labels_mu,
    "labels_alpha": labels_alpha,
    "labels_eps_homo": labels_eps_homo,
    "labels_eps_lumo": labels_eps_lumo,
    "labels_eps_gap": labels_eps_gap,
    "labels_elec_spatial_extent": labels_elec_spatial_extent,
    "labels_zpve": labels_zpve,
    "labels_U_0": labels_U_0,
    "labels_U": labels_U,
    "labels_H": labels_H,
    "labels_G": labels_G,
    "labels_C_v": labels_C_v,
}

data_out = {
    k: np.array(v)
    if "labels" in k or k in ["nr_atoms", "mol_ids"]
    else np.array(v, dtype=object)
    for k, v in data_out.items()
}

print("preprocessed dataset:")
print({k: v.shape for k, v in data_out.items()})

# save preprocessed dataset
print("Saving preprocessed dataset")
out_dir = "datasets/QM9"
os.makedirs(out_dir, exist_ok=True)
fname_out = out_dir + "/QM9_dataset_preprocessed.npz"
with open(fname_out, "wb") as f:
    np.savez(f, **data_out)


print("Processing data into padded dataset")

# add padding to dataset
## get max nr of atoms per type in the entire dataset
max_atoms_per_type = SortedDict()
for charge_array in data_out["atom_charges"]:
    atom_types, counts = np.unique(charge_array, return_counts=True)
    d = {atom_type: count for atom_type, count in zip(atom_types, counts)}
    for atom_type, count in d.items():
        if atom_type not in max_atoms_per_type:
            max_atoms_per_type[atom_type] = count
        else:
            max_atoms_per_type[atom_type] = max(max_atoms_per_type[atom_type], count)

print("max nr of atoms per atom charge: ", max_atoms_per_type)

atom_charges_for_masking = np.array(
    list(
        itertools.chain.from_iterable(
            [[c] * max_nr for c, max_nr in max_atoms_per_type.items()]
        )
    )
)

total_padding_length = sum([v for v in max_atoms_per_type.values()])
print("total size after padding: ", total_padding_length)
start_index_per_charge = {
    t: sum([v for k, v in max_atoms_per_type.items() if k < t])
    for t in max_atoms_per_type.keys()
}

n_points_total = len(atom_charges)
len_distances_padded = int(1 / 2 * total_padding_length * (total_padding_length - 1))

# initialize final padded arrays
atom_charges_padded = np.ndarray((n_points_total, total_padding_length))
coordinates_padded = np.ndarray((n_points_total, total_padding_length, 3))
distances_padded = np.ndarray((n_points_total, len_distances_padded))

# add padding per atom type
for i, (mol_atom_charges, mol_coordinates, mol_distances) in enumerate(
    zip(atom_charges, coordinates, distances)
):
    if i % 1000 == 0:
        print(i, "/", n_points_total)

    # sort in ascending order (same order as atom_charges_for_masking array)
    sorted_indices = np.argsort(mol_atom_charges)
    mol_atom_charges = mol_atom_charges[sorted_indices]
    mol_coordinates = mol_coordinates[sorted_indices, :]
    mol_distances = squareform(mol_distances)
    mol_distances = mol_distances[sorted_indices, :][:, sorted_indices]

    # collect segment masks per atom types
    mol_atom_charge_to_mask = {
        c: mol_atom_charges == c for c in np.unique(mol_atom_charges)
    }
    mol_atom_charge_pairs = list(
        itertools.product(np.unique(mol_atom_charges), repeat=2)
    )
    mol_atom_charge_pair_to_mask = {
        f"{p[0]}_{p[1]}": np.expand_dims(mol_atom_charge_to_mask[p[0]], 0).T
        * np.expand_dims(mol_atom_charge_to_mask[p[1]], 0)
        for p in mol_atom_charge_pairs
    }

    # initialize padding array with length the sum of all the max nr atoms per atom type
    ## padding for atom charges is 0.0
    mol_atom_charges_padded = np.zeros(total_padding_length)
    ## padding for distances is inf
    mol_distances_padded = np.full((total_padding_length, total_padding_length), np.inf)
    np.fill_diagonal(mol_distances_padded, 0.0)
    ## padding for coordinates is nan
    mol_coordinates_padded = np.full((total_padding_length, 3), np.nan)

    # add original values to padding arrays in correct locations
    ## charges and coordinates
    for c, mask in mol_atom_charge_to_mask.items():
        mol_original_charges = mol_atom_charges[mask]
        mol_original_coords = mol_coordinates[mask]
        start_idx = start_index_per_charge[c]
        mol_atom_charges_padded[
            start_idx : start_idx + len(mol_original_charges)
        ] = mol_original_charges
        mol_coordinates_padded[
            start_idx : start_idx + len(mol_original_coords), :
        ] = mol_original_coords

    ## distances
    for charge_pair, mask in mol_atom_charge_pair_to_mask.items():
        charge1, charge2 = [float(c) for c in charge_pair.split("_")]
        size1 = np.sum(mol_atom_charge_to_mask[charge1])
        size2 = np.sum(mol_atom_charge_to_mask[charge2])
        mol_original_distances = mol_distances[mask].reshape((size1, size2))

        start_idx1 = start_index_per_charge[charge1]
        start_idx2 = start_index_per_charge[charge2]

        mol_distances_padded[
            start_idx1 : start_idx1 + mol_original_distances.shape[0], :
        ][
            :, start_idx2 : start_idx2 + mol_original_distances.shape[1]
        ] = mol_original_distances

    mol_distances_padded = squareform(mol_distances_padded)

    # add padded arrays to final padded arrays
    atom_charges_padded[i] = mol_atom_charges_padded
    coordinates_padded[i] = mol_coordinates_padded
    distances_padded[i] = mol_distances_padded

data_out_padded = {
    "nr_atoms": data_out["nr_atoms"],
    "mol_ids": data_out["mol_ids"],
    "atom_charges": atom_charges_padded,
    "atom_charges_for_masking": atom_charges_for_masking,
    "coordinates": coordinates_padded,
    "distances": distances_padded,
}

data_out_padded.update({k: v for k, v in data_out.items() if "labels" in k})

print("data_out_padded:")
print({k: v.shape for k, v in data_out_padded.items()})


print("Saving data out padded")
# save padded dataset
fname_out_padded = out_dir + "/QM9_dataset_padded.npz"
with open(fname_out_padded, "wb") as f:
    np.savez(f, **data_out_padded)
