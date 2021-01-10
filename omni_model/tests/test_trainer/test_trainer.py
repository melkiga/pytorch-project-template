import pytest
from omni_model.src.trainer.base_trainer import BaseTrainer
from torch.nn import CrossEntropyLoss
from omni_model.tests.test_omni_model import optimizer, omnimodel
from omni_model.tests.test_data.test_datasets import cifar10dataset


@pytest.fixture
def basetrainer():
    yield BaseTrainer


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
