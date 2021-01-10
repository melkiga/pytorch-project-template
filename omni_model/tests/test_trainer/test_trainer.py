import pytest
from omni_model.src.trainer.base_trainer import BaseTrainer
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from omni_model.src.data.datasets import (
    _TRANSFORMS,
    _DATASET_TO_GROUP,
    _CIFAR10,
    _TRAIN,
)
from omni_model.tests.test_omni_model import resnet18, optimizer


@pytest.fixture
def basetrainer():
    yield BaseTrainer


class TestBaseTrainer:
    def test_default_trainer(self, basetrainer, resnet18, optimizer):
        resnet_model = resnet18(pretrained=True)
        basetrainer(
            net=resnet_model,
            optimizer=optimizer,
            criterion=CrossEntropyLoss(),
            data_loaders=datasets.CIFAR10(
                root="./data",
                train=True,
                transform=_TRANSFORMS[_DATASET_TO_GROUP[_CIFAR10]][_TRAIN],
            ),
        )
