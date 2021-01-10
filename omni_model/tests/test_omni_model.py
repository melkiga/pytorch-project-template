import pytest
import torch
from torchvision import models
from omni_model.src.omni_model import OmniModel


@pytest.fixture(scope="package")
def resnet18():
    yield models.resnet18


@pytest.fixture(scope="package")
def optimizer():
    yield torch.optim.SGD


@pytest.fixture
def omnimodel():
    yield OmniModel


class TestOmniModel:
    def test_default_omnimodel(self, omnimodel):
        omnimodel(model_arch="resnet18")

    def test_omnimodel_pretrained_imagenet(self, omnimodel):
        resnet = OmniModel(model_arch="resnet18", pretrained=True)
        resnet_layers = resnet.parameters()
        torch_resnet = models.resnet18(pretrained=True)
        torch_layers = torch_resnet.parameters()

        for r_layer, t_layer in zip(resnet_layers, torch_layers):
            assert torch.eq(r_layer, t_layer).all()
