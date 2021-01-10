import pytest
from omni_model.src.data.datasets import BaseDataset, ImageDataset
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
        BaseDataset(dataset_name="CIFAR10", subset_fraction=1.0)

    @pytest.mark.parametrize("subset_fraction", [1.0])
    @pytest.mark.parametrize("dataset_name", [*list(_SUPPORTED_DATASETS.keys())])
    def test_default_dataset(self, basedataset, dataset_name, subset_fraction):
        dataset = BaseDataset(
            dataset_name=dataset_name, subset_fraction=subset_fraction
        )
        assert dataset.dataset_root == _SUPPORTED_DATASETS[dataset_name]


@pytest.fixture
def imagedataset():
    yield ImageDataset


class TestImageDataset:
    def test_default_imagedataset_loads(self, imagedataset):
        dataset = ImageDataset(dataset_name="CIFAR10", subset_fraction=1.0)
        assert (
            dataset.transformation == _TRANSFORMS[_DATASET_TO_GROUP["CIFAR10"]][_VALID]
        )
