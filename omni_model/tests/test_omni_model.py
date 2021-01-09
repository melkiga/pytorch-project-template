import pytest
import torch
from torchvision import models


@pytest.fixture(scope="package")
def resnet18():
    yield models.resnet18


@pytest.fixture(scope="package")
def optimizer():
    yield torch.optim.SGD
