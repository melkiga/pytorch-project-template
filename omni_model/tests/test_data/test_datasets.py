import pytest
from omni_model.src.data.datasets import BaseDataset, ImageFolderDataset, CIFAR10Dataset
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

    @pytest.mark.parametrize("subset_fraction", [1.0])
    @pytest.mark.parametrize("dataset_name", [*list(_SUPPORTED_DATASETS.keys())])
    def test_default_dataset(self, basedataset, dataset_name, subset_fraction):
        dataset = basedataset(
            dataset_name=dataset_name, subset_fraction=subset_fraction
        )
        assert dataset.dataset_root == _SUPPORTED_DATASETS[dataset_name]


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
        assert dataset.images is not None
        assert len(dataset.class_names) == 10
        assert len(dataset) == 60000

    def test_cifar_getitem(self, cifar10dataset):
        dataset = cifar10dataset(
            dataset_name="CIFAR10", subset_fraction=1.0, transformation="DEFAULT"
        )
        i, (img, target) = enumerate(dataset).__next__()
        assert list(img.size()) == [3, 32, 32]
