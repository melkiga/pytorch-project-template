import pathlib
from typing import Tuple, Union
from torch.utils import data
import torchvision.transforms as transforms
from omni_model.src.utils.options import DatasetOptions
from omni_model.src import DATA_ROOT as _DATA_ROOT

_TRAIN = "train"
_VALID = "val"
_TEST = "test"

_CIFAR10 = "CIFAR10"

_SUPPORTED_DATASETS = {_CIFAR10: _DATA_ROOT / "cifar-10-batches-py"}

_DATASET_TO_GROUP = {_CIFAR10: "cifar"}

_TRANSFORMS = {
    "cifar": {
        _TRAIN: transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
        _VALID: transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
        _TEST: transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    }
}


class BaseDataset(data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        data_split: Tuple[int, int, int],
        subset_fraction: float,
        batch_size: int,
        num_workers: int,
    ):
        print(dataset_name)
        self.dataset_name = dataset_name
        self.data_split = data_split
        self.transforms = _TRANSFORMS[_DATASET_TO_GROUP[dataset_name]]
        self.dataset_root = _SUPPORTED_DATASETS[dataset_name]
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        pass

    def __getitem__(self):
        pass

    def split_data(self):
        pass

    def get_classes(self):
        pass

    def subset_data(self):
        pass


class RandomDataset(BaseDataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass


class ImageDataset(BaseDataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass


class CIFAR10Dataset(BaseDataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass


class CIFAR100Dataset(BaseDataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass
