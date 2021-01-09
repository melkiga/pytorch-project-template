import pytest
from omni_model.src.trainer.base_trainer import BaseTrainer
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from omni_model.src.datasets.data import transform
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
                root="./data", train=True, download=True, transform=transform
            ),
        )
