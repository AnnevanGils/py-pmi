import numpy as np
import os

fnames = ["../2CO2/data/2CO2_dataset1.npz", "../2CO2/data/2CO2_dataset2.npz"]

################# comments
# x: coordinates in bohr
#       in the order: C1_x, C1_y, C1_z, C2_x, C2_y, C2_z, O1_x, ... O4_z
# r: interatomic distances in angstrom (don't need this)

bohr_to_angstrom = 0.529177249
hartree_to_ev = 27.211396641308

# atoomgetallen
z_C = 6
z_O = 8

atom_charges = None
coordinates = None
distances = None
labels_CCSD = None
labels_HF = None
iterations = None


def build_stack(target, input):
    if target is None:
        return input
    else:
        print(f"stacking target ({target.shape}) and input ({input.shape})")
        return np.concatenate((target, input), axis=0)


for fname in fnames:
    print("----------------")
    with np.load(fname, allow_pickle=True) as f:
        for k in f:
            print(k, f[k].shape)
        print()

        len_dataset = f["x"].shape[0]

        # atom charges
        atom_charges = build_stack(
            atom_charges,
            np.tile(
                np.array([z_C, z_C, z_O, z_O, z_O, z_O], dtype=np.float16),
                (len_dataset, 1),
            ),
        )

        # coordinates
        coordinates = build_stack(
            coordinates,
            np.reshape(f["x"].astype(np.float64), (len_dataset, 6, 3))
            * bohr_to_angstrom,
        )

        # interatomic distances
        distances = build_stack(distances, f["r"])

        # labels CCSD
        labels_CCSD = build_stack(
            labels_CCSD, np.array([e["CCSD"] for e in f["E"]], dtype=np.float64)
        )

        # labels HF
        labels_HF = build_stack(
            labels_HF, np.array([e["HF"] for e in f["E"]], dtype=np.float64)
        )

        # iterations
        iterations = build_stack(iterations, f["it"])

    print()
    data_out = {
        "atom_charges": atom_charges,
        "coordinates": coordinates,
        "distances": distances,
        "labels_CCSD": labels_CCSD,
        "labels_HF": labels_HF,
        "iterations": iterations,
    }
    for k in data_out:
        print(k, data_out[k].shape, data_out[k].dtype)


# save combined dataset
out_dir = "datasets/2CO2"
os.makedirs(out_dir, exist_ok=True)
fname_out = out_dir + "/2CO2_dataset_combined.npz"
with open(fname_out, "wb") as f:
    np.savez(f, **data_out)
