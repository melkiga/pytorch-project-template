import pytest
import torch
from torchvision import models, transforms
from PIL import Image
from omni_model.src.omni_model import OmniModel


@pytest.fixture(scope="package")
def optimizer():
    yield torch.optim.SGD


@pytest.fixture
def omnimodel(scope="package"):
    yield OmniModel


class TestOmniModel:
    def test_default_omnimodel(self, omnimodel):
        omnimodel(model_arch="resnet18", pretrained=True)

    def test_omnimodel_pretrained_imagenet(self, omnimodel):
        resnet = omnimodel(model_arch="resnet18", pretrained=True)
        resnet_layers = resnet.parameters()
        torch_resnet = models.resnet18(pretrained=True)
        torch_layers = torch_resnet.parameters()
        for r_layer, t_layer in zip(resnet_layers, torch_layers):
            assert torch.eq(r_layer, t_layer).all()

    def test_omnimodel_freeze_layers(self, omnimodel):
        resnet = OmniModel(model_arch="resnet18", pretrained=True)
        resnet.freeze_layers()
        resnet_layers = resnet.parameters()
        for layer in resnet_layers:
            assert layer.requires_grad == False

    def test_omnimodel_predict(self, omnimodel):
        image = Image.open(
            "/home/melkiga/Documents/code/pytorch-project-template/data/n01495701_1216_ray.jpg"
        )
        label = "ray"
        label_num = 6
        transformation = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image = transformation(image)
        image = image.unsqueeze(0)
        resnet = omnimodel(model_arch="resnet18", pretrained=True)
        resnet.eval()
        outputs = resnet(image)
        assert max(list(outputs.shape)) == 1000

    @pytest.mark.parametrize("device", [0, torch.device(f"cuda:{0}")])
    def test_omnimodel_device(self, omnimodel, device):
        resnet = omnimodel(model_arch="resnet18", pretrained=True)
        resnet = resnet.cuda(device)
        assert resnet.device == torch.device("cuda:0")
        assert resnet.is_cuda == True
