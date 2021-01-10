import pathlib
from abc import ABC
from typing import Any, List, Tuple, Union, Optional, Callable
from PIL import Image
import pickle
import numpy as np
from torchvision import transforms
from omni_model.src.utils.options import DatasetOptions, TransformOptions
from omni_model.src.data.dataset_helpers import (
    _DATASET_TO_GROUP,
    _SUPPORTED_DATASETS,
    _TRANSFORMS,
    _TRAIN,
    _VALID,
    _CIFAR10,
    _EXAMPLE,
)
from omni_model.src.data.dataset_helpers import (
    check_integrity,
    download_and_extract_archive,
)


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
        transformation: TransformOptions = None,
        is_training: bool = False,
    ):
        self.dataset_name = dataset_name
        self.dataset_root = _SUPPORTED_DATASETS[dataset_name]
        self.subset_fraction = subset_fraction
        self.is_training = is_training
        phase_transform_type = _TRAIN if self.is_training else _VALID
        if transformation is not None:
            if transformation == "DEFAULT":
                self.transformation = _TRANSFORMS[_DATASET_TO_GROUP[dataset_name]][
                    phase_transform_type
                ]
            elif type(transformation) == transforms.Compose:
                self.transformation = transformation
            else:
                raise TypeError(f"Invalid transformation type {type(transformation)}.")

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def get_classes(self):
        raise NotImplementedError

    def subset_data(self):
        raise NotImplementedError

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


class BaseDataLoader(ABC):
    training_dataset: Optional[BaseDataset] = None
    testing_dataset: Optional[BaseDataset] = None
    validation_dataset: Optional[BaseDataset] = None

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


class ImageFolderDataset(BaseDataset):
    image_paths: List[pathlib.Path] = []
    images: List[Any] = []
    transformation: transforms = None
    __SUPPORTED_DATASETS = [_EXAMPLE]

    def __init__(
        self,
        dataset_name: str,
        subset_fraction: float,
        transformation: Optional[TransformOptions] = None,
        is_training: bool = BaseDataset.is_training,
    ):
        if dataset_name not in self.__SUPPORTED_DATASETS:
            raise ValueError(
                f"Invalid selection {dataset_name = }. Select from the following to create an ImageDataset: {', '.join(self.__SUPPORTED_DATASETS)}"
            )
        super().__init__(
            dataset_name=dataset_name,
            subset_fraction=subset_fraction,
            is_training=is_training,
            transformation=transformation,
        )

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
    images: Any

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
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
        subset_fraction: float,
        transformation: Optional[TransformOptions] = None,
        is_training: bool = BaseDataset.is_training,
        download: bool = False,
    ):
        super().__init__(
            dataset_name=_CIFAR10,
            subset_fraction=subset_fraction,
            is_training=is_training,
            transformation=transformation,
        )

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        if self.is_training:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        # now load the picked numpy arrays
        imgs = []
        for file_name, checksum in downloaded_list:
            file_path = self.dataset_root / file_name
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                imgs.append(entry["data"])
                if "labels" in entry:
                    self.labels.extend(entry["labels"])
                else:
                    self.labels.extend(entry["fine_labels"])

        self.images = np.vstack(imgs).reshape(-1, 3, 32, 32)
        self.images = self.images.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.images[index], self.labels[index]
        if self.transformation is not None:
            img = Image.fromarray(img)
            img = self.transformation(img)

        return img, target

    def _check_integrity(self) -> bool:
        root = self.dataset_root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = root / filename
            if not check_integrity(str(fpath), md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url, str(self.dataset_root), filename=self.filename, md5=self.tgz_md5
        )

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.is_training is True else "Test")


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
