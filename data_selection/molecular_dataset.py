import abc
import os

class MolDataset:
    def __init__(self, data_dir, download_dir, fname_preprocessed, fname_download, split, splits):
        path_preprocessed = self.get_path_preprocessed(data_dir, fname_preprocessed)
        path_downloaded = self.get_path_downloaded(download_dir, fname_download)

        if not(os.path.exists(path_preprocessed)):
            if not(os.path.exists(path_downloaded)):
                self.download(path_downloaded)
            
            self.preprocess(path_downloaded, path_preprocessed)

        if(splits == None):
            splits = self.generate_splits()
        
        self.idxs = splits[split]  

        self.atom_labels = None
        self.distances = None
        self.properties = None

    @abc.abstractmethod
    def download(self, path_downloaded) -> None:
        pass

    @abc.abstractmethod
    def preprocess(self, path_downloaded, path_preprocessed) -> None:
        pass
    
    @abc.abstractmethod
    def load_preprocessed(self, path_preprocessed):
        pass

    @abc.abstractmethod
    def generate_splits(self) -> dict:
        pass

    def get_path_downloaded(self, download_dir, fname_download):
        return os.path.join(os.getcwd(), download_dir, fname_download)
    
    def get_path_preprocessed(self, data_dir, fname_preprocessed):
        return os.path.join(os.getcwd(), data_dir, fname_preprocessed)