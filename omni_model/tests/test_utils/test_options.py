import pytest
from omni_model.src.utils.options import TransformOptions, TrainerOptions, ModelOptions
from torchvision.transforms import Compose
import torchvision.transforms as transforms


@pytest.fixture
def transform_option():
    yield TransformOptions


@pytest.fixture
def model_option():
    yield ModelOptions


@pytest.fixture
def trainer_option():
    yield TrainerOptions


@pytest.mark.parametrize(
    "transform", ["DEFAULT", None, Compose([transforms.RandomCrop(32, padding=4)])]
)
def test_transform_option(transform_option, transform):
    options: transform_option = {"transform": transform}
    print(options)


def test_trainer_option(trainer_option, model_arch="resnet18"):
    options: TrainerOptions = {"model_arch": "resnet18"}
    print(options)
