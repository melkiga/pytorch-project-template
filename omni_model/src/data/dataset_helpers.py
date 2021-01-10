import torchvision.transforms as transforms
from omni_model.src.utils.options import DatasetOptions
from omni_model.src import DATA_ROOT as _DATA_ROOT

_TRAIN = "train"
_VALID = "val"
_TEST = "test"

_CIFAR10 = "CIFAR10"
_EXAMPLE = "EXAMPLE"

_SUPPORTED_DATASETS = {
    _CIFAR10: _DATA_ROOT / "cifar-10-batches-py",
    _EXAMPLE: _DATA_ROOT / "",
}

_DATASET_TO_GROUP = {_CIFAR10: "cifar", _EXAMPLE: "example"}

_TRANSFORMS = {
    "example": {_TRAIN: {}, _VALID: {}},
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
    },
}
