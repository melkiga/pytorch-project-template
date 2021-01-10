import pytest
from omni_model.src.data.datasets import (
    BaseDataset,
    ImageFolderDataset,
    CIFAR10Dataset,
    BaseDataLoader,
)
from omni_model.src.data.dataset_helpers import (
    _SUPPORTED_DATASETS,
    _TRANSFORMS,
    _DATASET_TO_GROUP,
    _VALID,
)


@pytest.fixture
def basedataset():
    yield BaseDataset


class TestBaseDataset:
    def test_default_dataset_loads(self, basedataset):
        basedataset(dataset_name="EXAMPLE", subset_fraction=1.0)


@pytest.fixture
def imagedataset():
    yield ImageFolderDataset


class TestImageDataset:
    def test_default_imagedataset_loads(self, imagedataset):
        imagedataset(dataset_name="EXAMPLE", subset_fraction=1.0)


@pytest.fixture
def cifar10dataset(scope="package"):
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
def data_loader():
    yield BaseDataLoader


class TestBaseDataLoader:
    def test_default_loader(self, data_loader, basedataset):
        data_loader(basedataset(dataset_name="EXAMPLE", subset_fraction=1.0))

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