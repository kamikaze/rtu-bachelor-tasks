from pathlib import Path

import torch
import torch.utils.data

USE_CUDA = torch.cuda.is_available()


class BaseDataset(torch.utils.data.Dataset):
    DATASET_DIR_NAME = None
    METADATA_FILE_NAME = None

    def __init__(self, root: Path, transform=None, target_transform=None):
        super().__init__()
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')
        self.data_length = 0
        self.dataset_path = Path(root, self.DATASET_DIR_NAME)
        self.metadata_file_path = Path(self.dataset_path, 'metadata_fs.json')

    def _download(self, force: bool = False):
        pass

    def _prepare(self, force: bool = False):
        pass

    def load(self, force_download: bool = False, force_prepare: bool = False):
        self._download(force_download or force_prepare)
        self._prepare(force_prepare)

    def __len__(self):
        return self.data_length
