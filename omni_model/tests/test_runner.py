import pytest
import click
from omni_model.src.utils.options import (
    DatasetOptions,
    DeviceOptions,
    TrainerOptions,
    OptimizerOptions,
    _SUPPORTED_OPTIMIZERS,
)
from omni_model.src.runner import run

dataset_options: DatasetOptions = {
    "dataset_name": "CIFAR10",
    "subset_fraction": 0.5,
    "is_training": True,
    "data_split": (70, 20, 10),
    "num_workers": 8,
    "batch_size": 8,
}

device_options: DeviceOptions = {"use_gpu": True, "gpu_number": 0}
trainer_options: TrainerOptions = {
    "model_arch": "resnet18",
    "pretrained": True,
    "num_epochs": 5,
    "num_classes": 5,
}

optimizer_options: OptimizerOptions = {
    "optimizer": _SUPPORTED_OPTIMIZERS["SGD"],
    "learning_rate": 0.001,
}


def test_run():
    run(dataset_options, device_options, trainer_options, optimizer_options)
