import pathlib
from abc import ABC
from typing import Any, List, Tuple, Union
from torchvision import transforms
from omni_model.src.data.dataset_helpers import (
    _DATASET_TO_GROUP,
    _SUPPORTED_DATASETS,
    _TRANSFORMS,
)


class BaseDataset(ABC):
    samples: List[Any] = []
    labels: List[int] = []  # TODO: define set of types for labels (int, bbox, etc)
    class_names: List[str] = []
    is_training: bool = False

    def __init__(
        self,
        dataset_name: str,
        subset_fraction: float,
        is_training: bool = False,
    ):
        self.dataset_name = dataset_name
        self.dataset_root = _SUPPORTED_DATASETS[dataset_name]
        self.subset_fraction = subset_fraction
        self.is_training = is_training

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def get_classes(self):
        raise NotImplementedError

    def subset_data(self):
        raise NotImplementedError


class BaseDataLoader(ABC):
    training_dataset: BaseDataset = None
    testing_dataset: BaseDataset = None
    validation_dataset: BaseDataset = None

    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int,
        num_workers: int,
        data_split: Tuple[int, int, int],
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_split = data_split
        # TODO: split base dataset, return or set tr,te,val

    def split_data(self):
        raise NotImplementedError


class ImageDataset(BaseDataset):
    image_paths: List[pathlib.Path] = []
    images: List[Any] = []
    transformation: transforms = None

    def __init__(
        self,
        dataset_name: str,
        subset_fraction: float,
        is_training: bool = BaseDataset.is_training,
    ):
        super().__init__(
            dataset_name=dataset_name,
            subset_fraction=subset_fraction,
            is_training=is_training,
        )
        self.transformation = _TRANSFORMS[_DATASET_TO_GROUP[dataset_name]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        # convert to image (PIL)
        image = None

        # get label
        label = None

        if self.transformation:
            pass
            # transform image
        return (image, label)

    def get_classes(self):
        raise NotImplementedError

    def subset_data(self):
        raise NotImplementedError


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


class RandomDataset(BaseDataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass
