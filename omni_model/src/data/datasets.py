from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional, Callable
from PIL import Image
import pickle
import numpy as np
import random
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
from omni_model.src.data.dataset_helpers import (
    _DATASET_TO_GROUP,
    _SUPPORTED_DATASETS,
    _TRANSFORMS,
    _TRAIN,
    _VALID,
)
from omni_model.src.data.dataset_helpers import check_integrity


class BaseDataset(ABC):
    _repr_indent = 4
    samples: List[Any] = []
    labels: List[int] = []  # TODO: define set of types for labels (int, bbox, etc)
    class_names: List[str] = []
    is_training: bool = False
    transformation: Any = None

    def __init__(
        self,
        dataset_name: str,
        subset_fraction: float,
        transformation: transforms.Compose = None,
        is_training: bool = False,
    ):
        self._dataset_name = dataset_name
        self._dataset_root = _SUPPORTED_DATASETS[dataset_name]
        self._subset_fraction = subset_fraction
        self._is_training = is_training
        self._samples = []
        self._labels = []
        self._class_names = []
        phase_transform_type = _TRAIN if self.is_training else _VALID
        if transformation is not None:
            if transformation == "DEFAULT":
                self._transformation = _TRANSFORMS[_DATASET_TO_GROUP[dataset_name]][
                    phase_transform_type
                ]
            elif type(transformation) == transforms.Compose:
                self._transformation = transformation
            else:
                raise TypeError(f"Invalid transformation type {type(transformation)}.")

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, samples):
        self._samples = samples

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels

    @property
    def class_names(self):
        return self._class_names

    @class_names.setter
    def class_names(self, class_names):
        self._class_names = class_names

    @property
    def dataset_root(self):
        return self._dataset_root

    @dataset_root.setter
    def dataset_root(self, dataset_root):
        self._dataset_root = dataset_root

    @property
    def subset_fraction(self):
        return self._subset_fraction

    @subset_fraction.setter
    def subset_fraction(self, subset_fraction):
        self._subset_fraction = subset_fraction

    @property
    def transformation(self):
        return self._transformation

    @transformation.setter
    def transformation(self, transformation):
        self._transformation = transformation

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def get_classes(self):
        raise NotImplementedError

    def subset_data(self):
        if self.subset_fraction < 1:
            sample_indexes = range(len(self._labels))
            subset_size = int(len(self._labels) * self.subset_fraction)
            sampled_indexes = random.sample(sample_indexes, subset_size)
            self.samples = [self.samples[index] for index in sampled_indexes]
            self.labels = [self.labels[index] for index in sampled_indexes]

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.dataset_root is not None:
            body.append("Root location: {}".format(self.dataset_root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transformation") and self.transformation is not None:
            body += [repr(self.transformation)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return ["{}{}".format(head, lines[0])] + [
            "{}{}".format(" " * len(head), line) for line in lines[1:]
        ]

    def extra_repr(self) -> str:
        return ""


class DataLoaderWrapper:
    training_dataloader: Optional[DataLoader] = None
    validation_dataloader: Optional[DataLoader] = None
    testing_dataloader: Optional[DataLoader] = None

    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int = 32,
        num_workers: int = 8,
        data_split: Tuple[int, int, int] = (100, 0, 0),
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_split = data_split
        training_dataset, validation_dataset, testing_dataset = self.split_data(dataset)

        self.training_dataloader = DataLoader(
            training_dataset, batch_size=batch_size, num_workers=num_workers
        )
        self.validation_dataloader = DataLoader(
            validation_dataset, batch_size=batch_size, num_workers=num_workers
        )
        self.testing_dataloader = DataLoader(
            testing_dataset, batch_size=batch_size, num_workers=num_workers
        )

    def split_data(self, dataset):
        split_sizes = tuple(
            int(split / 100 * len(dataset)) for split in self.data_split
        )

        (
            training_dataset,
            validation_dataset,
            testing_dataset,
        ) = random_split(dataset, split_sizes)
        return training_dataset, validation_dataset, testing_dataset


class CIFAR10Dataset(BaseDataset):
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]
    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        dataset_name: str,
        subset_fraction: float,
        transformation: Optional[transforms.Compose] = None,
        is_training: bool = BaseDataset.is_training,
    ):
        super().__init__(
            dataset_name=dataset_name,
            subset_fraction=subset_fraction,
            is_training=is_training,
            transformation=transformation,
        )

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " Please download the dataset first using the download-data command."
            )

        downloaded_list = self.train_list + self.test_list

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = self.dataset_root / file_name
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.samples.append(entry["data"])
                if "labels" in entry:
                    self.labels.extend(entry["labels"])
                else:
                    self.labels.extend(entry["fine_labels"])

        self.samples = np.vstack(self.samples).reshape(-1, 3, 32, 32)
        self.samples = self.samples.transpose((0, 2, 3, 1))  # convert to HWC

        if subset_fraction < 1.0:
            self.subset_data()

        self.num_classes = self.get_classes()

        if self.num_classes != len(set(self.labels)):
            self.num_classes = len(set(self.labels))

    def get_classes(self) -> None:
        path = self.dataset_root / self.meta["filename"]
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted."
                + " You can use download=True to download it"
            )
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.class_names = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.class_names)}
        return len(self.class_names)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.samples[index], self.labels[index]
        if self.transformation is not None:
            img = Image.fromarray(img)
            img = self.transformation(img)

        return img, target

    def _check_integrity(self) -> bool:
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = self.dataset_root / filename
            if not check_integrity(str(fpath), md5):
                return False
        return True

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.is_training is True else "Test")

    def subset_data(self):
        super().subset_data()
        self.samples = np.vstack(self.samples).reshape(-1, 32, 32, 3)
