import pytest
from omni_model.src.utils.options import TransformOptions
from torchvision.transforms import Compose
import torchvision.transforms as transforms


@pytest.fixture
def transform_option():
    yield TransformOptions


@pytest.mark.parametrize(
    "transform", ["DEFAULT", None, Compose([transforms.RandomCrop(32, padding=4)])]
)
def test_transform_option(transform_option, transform):
    options: transform_option = {"transform": transform}
    print(options)
