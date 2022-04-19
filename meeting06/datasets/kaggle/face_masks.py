import json
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data
import torch.utils.data
from PIL import Image, UnidentifiedImageError
from numpy.lib.format import open_memmap

from meeting06.datasets.kaggle import KaggleDataset
from meeting06.datasets.utils import adjust_image


class DatasetMasks(KaggleDataset):
    DATASET_NAME = [
        'tapakah68/medical-masks-part1',
        'tapakah68/medical-masks-part2',
    ]
    BYTES_PER_VALUE = 16 / 8
    DATASET_DIR_NAME = 'masks'
    IMAGE_DIR_NAME = 'images'
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    RESULT_FILE_NAME = 'df.csv'
    LABELS = {
        0: 'The mask is worn correctly, covers the nose and mouth.',
        1: 'The mask covers the mouth, but does not cover the nose.',
        2: 'The mask is on, but does not cover the nose or mouth.',
        3: 'There is no mask on the face.',
    }

    def __init__(self, root: Path, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)

        if self.METADATA_FILE_NAME:
            self.metadata_file_path = Path(self.dataset_path, self.METADATA_FILE_NAME)
        else:
            self.metadata_file_path = None

    def _download(self, force: bool = False):
        if force or not self.dataset_path.exists():
            super()._download(force)

            second_file_path = Path(self.dataset_path, 'df_part_2.csv')

            with open(second_file_path) as fi, open(Path(self.dataset_path, 'df.csv'), 'a') as fo:
                next(fi)
                fo.writelines(fi)

            second_file_path.unlink(missing_ok=True)

    @staticmethod
    def _iter_images(root: Path, min_size: Optional[int] = None):
        # ID, TYPE, USER_ID, GENDER, AGE, name, size_mb

        for file_item in root.iterdir():
            if file_item.is_file():
                try:
                    with Image.open(file_item) as image:
                        if min_size and min(image.size) < min_size:
                            continue

                        _type = file_item.name.split('_', maxsplit=2)[1]

                        yield _type, image
                except UnidentifiedImageError:
                    pass

    def _get_min_size(self, limit_size: Optional[int] = None):
        min_size = 999999

        for _, image in self._iter_images(self.image_dir_path, limit_size):
            width, height = image.size

            # Filter out images with any dimension smaller than self.IMAGE_WIDTH px
            min_size = max(limit_size, min(min_size, min(height, width)))

        return min_size

    @staticmethod
    @abstractmethod
    def _resize_images(size: int):
        pass

    def _prepare(self, force: bool = True):
        if force or not self.metadata_file_path.exists():
            min_size = self._get_min_size()
            print(f'Found minimum size: {min_size}px')

            image_set = self._resize_images(min_size)
            directories = sorted(
                file_item.name
                for file_item in filter(lambda i: i.is_dir() and i.name[0] != '.', self.dataset_path.iterdir())
            )

            metadata = {
                'labels': {idx: dir_name for idx, dir_name in enumerate(directories)},
                'images': image_set
            }

            with open(self.metadata_file_path, 'w') as fm:
                json.dump(metadata, fm)


class DatasetMasksNumpyMmap(DatasetMasks):
    DATASET_DTYPE = 'float16'
    METADATA_FILE_NAME = 'metadata_numpy_mmap.json'

    def __init__(self, root: Path, transform=None, target_transform=None):
        super().__init__(root, transform, target_transform)
        self.dataset_file_path = Path(self.dataset_path, 'data.npy')

    def __count_images(self, min_size: Optional[int] = None) -> int:
        cnt = sum(
            int(min(image.width, image.height) >= min_size)
            for _, image in self._iter_images(self.image_dir_path)
        )

        return cnt

    def _resize_images(self, size: Optional[int] = None):
        item_count = self.__count_images(size)
        print(f'Item count: {item_count}')

        dataset_shape = (item_count, 3, size, size)

        x = open_memmap(
            str(self.dataset_file_path), mode='w+', dtype=self.DATASET_DTYPE, shape=dataset_shape
        )
        y = []

        for idx, (label, image) in enumerate(self._iter_images(self.image_dir_path, size)):
            file_name = Path(image.filename).name
            print(f'{idx}/{item_count}: Processing {file_name}')

            try:
                image = adjust_image(image, size, size)
            except OSError as e:
                print(f'Failed reading image file: {file_name}, {e}. Deleting original file')
                Path(image.filename).unlink(missing_ok=True)
            else:
                image = np.asarray(image) / 255.0
                # Converting HxWxC to CxHxW
                image = np.transpose(image, (2, 0, 1))
                x[idx, :, :] = image[:, :]
                # Decreasing by 1 due to indexes start from 1
                y.append(int(label) - 1)

        x.flush()

        metadata = {
            'shape': (len(y), 3, size, size),
            'labels': y
        }

        return metadata

    def _prepare(self, force: bool = True):
        if force or not self.dataset_file_path.exists() or not self.metadata_file_path.exists():
            print('Preparing dataset')
            min_size = self._get_min_size(self.IMAGE_WIDTH)
            print(f'Found minimum size: {min_size}px')

            metadata = self._resize_images(min_size)

            with open(self.metadata_file_path, 'w') as fm:
                json.dump(metadata, fm)

    def load(self, force_download: bool = False, force_prepare: bool = False):
        super().load(force_download, force_prepare)

        with open(self.metadata_file_path) as f:
            metadata = json.load(f)

        self.dataset_shape = metadata['shape']
        self.data_length = self.dataset_shape[0]
        self.y = metadata['labels']

        self.x = open_memmap(
            str(self.dataset_file_path), mode='r', dtype=self.DATASET_DTYPE, shape=self.dataset_shape
        )
        self.y = F.one_hot(torch.tensor(self.y, dtype=torch.long))

#    def __len__(self):
#        return 10000

    def __getitem__(self, index):
        x = self.x[index]
        x = np.array(x)
        x = torch.from_numpy(x)

        return x, self.y[index]
