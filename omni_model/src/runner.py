# TODO: https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
from omni_model.src.utils.options import (
    DatasetOptions,
    DeviceOptions,
    TrainerOptions,
    ModelOptions,
    DataLoaderOptions,
    OptimizerOptions,
)
from omni_model.src.data.datasets import CIFAR10Dataset, DataLoaderWrapper
from omni_model.src.trainer.omni_trainer import OmniTrainer
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
    optimizer_options: OptimizerOptions = None,
):

    # TODO: load parameters

    # set torch device
    if device_options["use_gpu"]:
        torch.backends.cudnn.benchmark = True
    device = torch.device(
        f"cuda:{device_options['gpu_number']}" if device_options["use_gpu"] else "cpu"
    )

    # load the dataset and data loader
    data_loader_options: DataLoaderOptions = {}
    for key in DataLoaderOptions.__optional_keys__:
        if key in dataset_options:
            data_loader_options[key] = dataset_options.pop(key)

    if dataset_options["dataset_name"] == "CIFAR10":
        dataset = CIFAR10Dataset(**dataset_options)
    else:
        raise NotImplementedError  # TODO: load dataset
    data_loader_wrapper = DataLoaderWrapper(dataset=dataset, **data_loader_options)

    # TODO: load model
    model_options: ModelOptions = {}
    for key in ModelOptions.__required_keys__:
        if key not in trainer_options:
            raise ValueError("")
        model_options[key] = trainer_options.pop(key)
    for key in ModelOptions.__optional_keys__:
        if key in trainer_options:
            model_options[key] = trainer_options.pop(key)
    model = OmniModel(**model_options, device=device, num_classes=dataset.num_classes)
    if device_options["use_gpu"]:
        model = model.cuda(device)

    optim = optimizer_options["optimizer"](
        params=model.parameters(), lr=optimizer_options["learning_rate"]
    )

    # setup training params
    trainer = OmniTrainer(
        model=model,
        data_loaders=data_loader_wrapper,
        optimizer=optim,
        criterion=torch.nn.CrossEntropyLoss(),
        **trainer_options,
    )

    # setup logger

    # setup metrics

    # trainer.tune()

    # TODO: set up a visualizer for pretty plots

    # setup output directory