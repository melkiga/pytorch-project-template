import pytest
from omni_model.src.utils.options import TrainerOptions, ModelOptions
from torchvision.transforms import Compose
import torchvision.transforms as transforms


@pytest.fixture
def model_option():
    yield ModelOptions


@pytest.fixture
def trainer_option():
    yield TrainerOptions


def test_trainer_option(trainer_option, model_arch="resnet18"):
    options: TrainerOptions = {"model_arch": "resnet18"}
    print(options)
