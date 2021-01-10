import pathlib
from abc import ABC
from typing import Any, List, Tuple, Union, Optional, Callable
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


class BaseDataset(ABC):
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
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

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
    ):
        super().__init__(
            dataset_name=_CIFAR10,
            subset_fraction=subset_fraction,
            is_training=is_training,
            transformation=transformation,
        )

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
