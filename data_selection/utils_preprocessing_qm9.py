import os
import urllib.request
import logging
import tarfile
import numpy as np
from scipy.spatial.distance import pdist

def get_fname_download_qm9():
    return 'dsgdb9nsd.xyz.tar.bz2'

def get_fname_preprocessed_qm9():
    return "qm9_preprocessed.npz"

def download_qm9_data(path_download):
    url = 'https://springernature.figshare.com/ndownloader/files/3195389'
        
    os.makedirs(path_download)

    logging.info(f'Downloading QM9 dataset. Files will be in directory: {os.path.dirname(path_download)}')
    print(f'Downloading QM9 dataset. Files will be in directory: {os.path.dirname(path_download)}')

    urllib.request.urlretrieve(url, filename=path_download)

    logging.info('QM9 dataset downloaded successfully.')
    print('QM9 dataset downloaded successfully.')    

def process_xyz_qm9(datafile):

    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]
    
    n_atoms = int(xyz_lines[0].strip())
    
    # molecular properties 'gdb tag', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv' (gdb tag is skipped)
    mol_properties_keys = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    mol_properties = [float(n) for n in xyz_lines[1].strip().split('\t')[1:]]

    # per atom label, a x, y, z position (in Å) and Mulliken partial charge (in e)
    atom_labels, coordinates, partial_charges = [], np.zeros((n_atoms, 3)), []

    for i in range(n_atoms):
        line_items = xyz_lines[2+i].replace('*^', 'e').strip().split('\t')
        atom_labels.append(line_items[0])
        # comment: float might not be sufficient accuracy for atom xyz positions
        coordinates[i,:] = np.array([float(n) for n in line_items[1:-1]])
        partial_charges.append(float(line_items[-1]))

    atom_labels = np.array(atom_labels)
    partial_charges = np.array(partial_charges)  

    # calculate distances from coordinates
    distances = pdist(coordinates)

    # harmonic vibrational frequencies
    frequencies = np.array([float(n) for n in xyz_lines[n_atoms+2].strip().split('\t')])

    # SMILES strings from GDB-17 and from B3LYP relaxation
    smiles = np.array([s for s in xyz_lines[n_atoms+3].strip().split('\t')])

    # InChI strings for Corina and B3LYP geometries
    inchi = np.array([s for s in xyz_lines[n_atoms+4].strip().split('\t')])

    molecule =  {'nr_atoms': n_atoms, 'atom_labels': atom_labels, 'coordinates': coordinates, 'distances': distances, 'partial_charges': partial_charges, 'frequencies': frequencies, 'smiles': smiles, 'inchi': inchi}
    molecule.update({k: v for k, v in zip(mol_properties_keys, mol_properties)})

    return molecule


def preprocess_qm9_data(path_downloaded, path_preprocessed):
    
    print("loading raw data...")

    if(tarfile.is_tarfile(path_downloaded)):
        tardata = tarfile.open(path_downloaded, 'r')
        files = tardata.getmembers()

    else:
        raise ValueError('file should be a tar file or otherwise tardata object')


    print("starting preprocessing...")

    molecules = []

    for file in files:
        with tardata.extractfile(file) as openfile:
            molecules.append(process_xyz_qm9(openfile))

    print(f"processed {len(molecules)} molecules")
    
    # Check that all molecules have the same set of items in their dictionary:
    props = molecules[0].keys()
    assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys'


    # Convert list-of-dicts to dict-of-lists
    molecules = {prop: np.array([mol[prop] for mol in molecules]) for prop in props}

    print("done preprocessing")

    # save arrays with molecular data to disk
    if not(os.path.exists(os.path.dirname(path_preprocessed))):
        os.makedirs(path_preprocessed)

    np.savez(path_preprocessed, **molecules)

def get_qm9_splits(download_dir):
    def get_excl_idxs(download_dir, data_path_txt):
        def is_int(str):
            try:
                int(str)
                return True
            except:
                return False
        
        # file location for pickled list
        path_cached = os.path.join(download_dir, 'excl_idxs.npy')

        if(os.path.exists(path_cached)):
            with open(path_cached, 'rb') as f:
                return np.load(f)
        else:
            with open(data_path_txt, 'r') as f:
                lines = f.readlines()
                excluded_strings = [line.split()[0]
                                    for line in lines if len(line.split()) > 0]
            
            excl_idxs = np.array([int(idx) - 1 for idx in excluded_strings if is_int(idx)], dtype=np.int)

            # save for reuse later
            with open(path_cached, 'wb') as f:
                np.save(f, excl_idxs)

            return excl_idxs
    
    if not(os.path.exists(download_dir)):
        os.makedirs(download_dir)

    # check if file 'uncharacterized.txt' is in download dir yet
    # if not, download it
    data_path = os.path.join(download_dir, 'uncharacterized.txt')
    if not(os.path.exists(data_path)):
        url = 'https://springernature.figshare.com/ndownloader/files/3195404'
        urllib.request.urlretrieve(url, filename=data_path)
    
    excl_idxs = get_excl_idxs(download_dir, data_path)

    # remove excluded indices and make splits
    Ngdb9 = 133885
    Nexcluded = 3054

    assert len(excl_idxs) == Nexcluded, f'There should be exactly {Nexcluded} excluded molecules. Found {len(excl_idxs)}'

    included_idxs = np.array(
    sorted(list(set(range(Ngdb9)) - set(excl_idxs))))

    # Now generate random permutations to assign molecules to training/validation/test sets.
    Nmols = Ngdb9 - Nexcluded

    Ntrain = 100000
    Ntest = int(0.1*Nmols)
    Nvalid = Nmols - (Ntrain + Ntest)

    # Generate random permutation
    np.random.seed(0)
    data_perm = np.random.permutation(Nmols)

    # Now use the permutations to generate the indices of the dataset splits.
    # train, valid, test, extra = np.split(included_idxs[data_perm], [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])
    train, valid, test, extra = np.split(
        data_perm, [Ntrain, Ntrain+Nvalid, Ntrain+Nvalid+Ntest])

    assert(len(extra) == 0), 'Split was inexact {} {} {} {}'.format(
        len(train), len(valid), len(test), len(extra))

    return {'train': included_idxs[train], 'test': included_idxs[test], 'validate': included_idxs[valid]}


    

    
    

    
    

    