import pytest
from omni_model.src.data.datasets import (
    CIFAR10Dataset,
    DataLoaderWrapper,
)
from omni_model.src.data.dataset_helpers import (
    _SUPPORTED_DATASETS,
    _TRANSFORMS,
    _DATASET_TO_GROUP,
    _VALID,
)
from omni_model.src.utils.options import (
    DatasetOptions,
    DataLoaderOptions,
)

dataset_options: DatasetOptions = {
    "dataset_name": "CIFAR10",
    "subset_fraction": 0.5,
    "is_training": True,
}

dataloader_options: DataLoaderOptions = {
    "data_split": (70, 20, 10),
    "num_workers": 8,
    "batch_size": 8,
}


def test_cifar_data_loading_options(cifar10dataset, data_loader):
    dataset = cifar10dataset(**dataset_options)
    assert len(dataset) == 30000
    data_loaders = data_loader(dataset, **dataloader_options)
    split_sizes = tuple(
        int(split / 100 * len(dataset)) for split in dataloader_options["data_split"]
    )
    assert len(data_loaders.training_dataloader.dataset) == split_sizes[0]
    assert len(data_loaders.validation_dataloader.dataset) == split_sizes[1]
    assert len(data_loaders.testing_dataloader.dataset) == split_sizes[2]


@pytest.fixture
def cifar10dataset():
    yield CIFAR10Dataset


class TestCIFAR10Dataset:
    def test_default_cifar10dataset_loads(self, cifar10dataset):
        dataset = cifar10dataset(
            dataset_name="CIFAR10",
            subset_fraction=1.0,
            transformation="DEFAULT",
        )
        assert (
            dataset.transformation == _TRANSFORMS[_DATASET_TO_GROUP["CIFAR10"]][_VALID]
        )
        assert dataset.samples is not None
        assert len(dataset.class_names) == 10
        assert len(dataset) == 60000

    def test_cifar_getitem(self, cifar10dataset):
        dataset = cifar10dataset(
            dataset_name="CIFAR10", subset_fraction=1.0, transformation="DEFAULT"
        )
        i, (img, target) = enumerate(dataset).__next__()
        assert list(img.size()) == [3, 32, 32]

    def test_cifar_dataset_subset(self, cifar10dataset):
        dataset = cifar10dataset(dataset_name="CIFAR10", subset_fraction=0.5)
        assert len(dataset) == 30000


@pytest.fixture
def data_loader(scope="package"):
    yield DataLoaderWrapper


class TestBaseDataLoader:
    def test_default_loader(self, data_loader, cifar10dataset):
        data_loader(cifar10dataset(dataset_name="CIFAR10", subset_fraction=1.0))

    @pytest.mark.parametrize("data_split", [(40, 30, 30), (98, 1, 1), (80, 10, 10)])
    def test_loader_nonempty_splits(self, data_loader, cifar10dataset, data_split):
        dataset = cifar10dataset(
            dataset_name="CIFAR10", subset_fraction=0.5, transformation="DEFAULT"
        )
        data_loaders = data_loader(dataset, data_split=data_split)
        split_sizes = tuple(int(split / 100 * len(dataset)) for split in data_split)

        assert len(data_loaders.training_dataloader.dataset) == split_sizes[0]
        assert len(data_loaders.validation_dataloader.dataset) == split_sizes[1]
        assert len(data_loaders.testing_dataloader.dataset) == split_sizes[2]

    def test_loader_empty_splits(self, data_loader, cifar10dataset):
        dataset = cifar10dataset(
            dataset_name="CIFAR10", subset_fraction=0.5, transformation="DEFAULT"
        )
        data_loaders = data_loader(dataset, data_split=(50, 50, 0))
        split_sizes = tuple(int(split / 100 * len(dataset)) for split in (50, 50, 0))

        assert len(data_loaders.training_dataloader.dataset) == split_sizes[0]
        assert len(data_loaders.validation_dataloader.dataset) == split_sizes[1]
        assert len(data_loaders.testing_dataloader.dataset) == split_sizes[2]
