import pytest
import click
from omni_model.src.utils.options import DatasetOptions, DeviceOptions, TrainerOptions
from omni_model.src.runner import run


def test_run():
    dataset_options: DatasetOptions = {
        "dataset_name": "CIFAR10",
        "subset_fraction": 1.0,
    }

    device_options: DeviceOptions = {"use_gpu": True, "gpu_number": 0}
    trainer_options: TrainerOption = {
        "model_arch": "resnet18",
        "pretrained": True,
    }

    run(dataset_options, device_options, trainer_options)