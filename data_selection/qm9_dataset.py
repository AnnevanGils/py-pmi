import os
import numpy as np
from utils_preprocessing_qm9 import download_qm9_data, preprocess_qm9_data, get_fname_preprocessed_qm9, get_fname_download_qm9, get_qm9_splits
from molecular_dataset import MolDataset


class QM9_dataset(MolDataset):
    def __init__(self, data_dir, download_dir, split, splits=None, properties='all'):
        fname_download = get_fname_download_qm9()
        fname_preprocessed = get_fname_preprocessed_qm9()
        data_dir = os.path.join(data_dir, 'qm9')
        download_dir = os.path.join(download_dir, 'qm9')
        self._download_dir = download_dir

        super().__init__(data_dir, download_dir, fname_preprocessed, fname_download, split, splits)

        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}

        path_preprocessed = self.get_path_preprocessed(data_dir, fname_preprocessed)

        data = self.load_preprocessed(path_preprocessed)      

        self.nr_atoms = data['nr_atoms'][self.idxs]
        self.atom_labels = data['atom_labels'][self.idxs]
        self.distances = data['distances'][self.idxs]

        if(properties == "all"):
            properties = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
        
        self.properties = {prop: self.transform_unit(prop, data[prop], qm9_to_eV)[self.idxs] for prop in properties}

    
    def download(self, path_downloaded) -> None:
        download_qm9_data(path_downloaded)

    def preprocess(self, path_downloaded, path_preprocessed) -> None:
        preprocess_qm9_data(path_downloaded, path_preprocessed)
    
    def load_preprocessed(self, path_preprocessed):
        return np.load(path_preprocessed, allow_pickle=True)

    def generate_splits(self) -> dict:
        return get_qm9_splits(self._download_dir)   

    def transform_unit(self, prop_key, data, transformation_table):
        if(prop_key in transformation_table):
            return transformation_table[prop_key]*data
        else:
            return data
