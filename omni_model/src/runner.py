# TODO: https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
from omni_model.src.utils.options import (
    DatasetOptions,
    DeviceOptions,
    TrainerOptions,
    ModelOptions,
)
from omni_model.src.data.datasets import CIFAR10Dataset
from omni_model.src.trainer.base_trainer import BaseTrainer
from omni_model.src.data.dataset_helpers import download_and_extract_archive
from omni_model.src.omni_model import OmniModel
from typing import Optional
import torch


def download(
    dataset_root: str,
    url: str,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
) -> None:
    # TODO: check if data exists in root.
    download_and_extract_archive(url, dataset_root, filename=filename, md5=md5)


def run(
    dataset_options: DatasetOptions = None,
    device_options: DeviceOptions = None,
    trainer_options: TrainerOptions = None,
):

    # TODO: load parameters

    # set torch device
    if device_options["use_gpu"]:
        torch.backends.cudnn.benchmark = True
    device = torch.device(
        f"cuda:{device_options['gpu_number']}" if device_options["use_gpu"] else "cpu"
    )

    # engine can train, eval, optimize the model
    # engine can save and load the model and optimizer

    if dataset_options["dataset_name"] == "CIFAR10":
        dataset = CIFAR10Dataset(**dataset_options)
    else:
        raise NotImplementedError  # TODO: load dataset
    # dataset is a dictionary that contains all the needed datasets indexed by modes
    # (example: dataset.keys() -> ['train','eval'])

    # TODO: load model
    # TODO: initialize engine
    model_options: ModelOptions = {}
    for key in ModelOptions.__required_keys__:
        if key not in trainer_options:
            raise ValueError("")
        model_options[key] = trainer_options.pop(key)
    for key in ModelOptions.__optional_keys__:
        if key in trainer_options:
            model_options[key] = trainer_options.pop(key)
    model = OmniModel(**model_options)

    # model includes a network, a criterion and a metric
    # model can register engine hooks (begin epoch, end batch, end batch, etc.)
    # (example: "calculate mAP at the end of the evaluation epoch")
    # note: model can access to datasets using engine.dataset

    # setup training params
    trainer = BaseTrainer(model=model, **trainer_options)

    # setup logger

    # setup metrics

    # TODO: set up a visualizer for pretty plots

    # setup output directory