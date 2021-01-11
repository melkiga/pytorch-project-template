import pytest
from omni_model.src.omni_model import OmniModel
from omni_model.src.data.datasets import CIFAR10Dataset, DataLoaderWrapper
from omni_model.src.trainer.base_trainer import BaseTrainer
from omni_model.src.trainer.omni_trainer import OmniTrainer
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from PIL import Image
import torch
from torchvision import transforms

transformation = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
image = Image.open(
    "/home/melkiga/Documents/code/pytorch-project-template/data/n01495701_1216_ray.jpg"
)
image = transformation(image).unsqueeze(0)
label = torch.tensor([6]).long()


@pytest.fixture
def basetrainer():
    yield BaseTrainer


@pytest.fixture
def omnimodel():
    yield OmniModel(model_arch="resnet18", pretrained=True)


@pytest.fixture
def optimizer():
    yield SGD


@pytest.fixture
def data_loaders():
    yield DataLoaderWrapper(
        CIFAR10Dataset(
            dataset_name="CIFAR10",
            subset_fraction=0.1,
            is_training=True,
            transformation="DEFAULT",
        ),
        batch_size=8,
        num_workers=8,
        data_split=(100, 0, 0),
    )


class TestBaseTrainer:
    def test_default_trainer(self, basetrainer, omnimodel, optimizer, cifar10dataset):
        resnet_model = omnimodel(model_arch="resnet18", pretrained=True)
        dataset = cifar10dataset(
            dataset_name="CIFAR10",
            subset_fraction=1.0,
            is_training=True,
            transformation="DEFAULT",
        )
        basetrainer(
            model=resnet_model,
            optimizer=optimizer,
            criterion=CrossEntropyLoss(),
            data_loaders=dataset,
        )


@pytest.fixture
def omnitrainer():
    yield OmniTrainer


class TestOmniTrainer:
    def test_default_omni_trainer(
        self, omnitrainer, omnimodel, optimizer, data_loader, cifar10dataset
    ):
        resnet_model = omnimodel(model_arch="resnet18", pretrained=True)
        optim = optimizer(resnet_model.parameters(), lr=0.001)
        data_loader = data_loader(
            cifar10dataset(
                dataset_name="CIFAR10",
                subset_fraction=1.0,
                is_training=True,
                transformation="DEFAULT",
            ),
            batch_size=8,
            num_workers=1,
        )

        trainer = omnitrainer(
            model=resnet_model,
            optimizer=optim,
            criterion=CrossEntropyLoss(),
            data_loaders=data_loader,
            num_epochs=1,
        )

    def test_omni_trainer_train_step(
        self, omnitrainer, omnimodel, optimizer, data_loaders
    ):
        omnimodel.train()
        optim = optimizer(omnimodel.parameters(), lr=0.001)
        trainer = omnitrainer(
            model=omnimodel,
            optimizer=optim,
            criterion=CrossEntropyLoss(),
            data_loaders=data_loaders,
            num_epochs=1,
        )

        trainer.train_step(image, label)
