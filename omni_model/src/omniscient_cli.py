import click
from omni_model.src.cli_helpers import (
    validate_dataset_choice,
    validate_data_split,
    validate_num_workers,
    PythonLiteralOption,
    validate_device,
    validate_dataset_root,
    NotRequiredIf,
    _DOWNLOAD_DATA_URLS,
)
from omni_model.src.omni_model import model_names as _SUPPORTED_MODEL_ARCHS
from omni_model.src.data.dataset_helpers import _SUPPORTED_DATASETS
from omni_model.src.utils.options import (
    DatasetOptions,
    DeviceOptions,
    TrainerOptions,
    OptimizerOptions,
    _SUPPORTED_OPTIMIZERS,
)
from omni_model.src.runner import run, download


@click.group()
def omniscient_cli():
    pass


@click.group()
def omniscient_datasets():
    pass


@omniscient_datasets.command()
@click.option(
    "--dataset-root",
    type=str,
    callback=validate_dataset_root,
    help="Path to download data.",
)
@click.option(
    "--dataset-name",
    type=click.Choice([*_DOWNLOAD_DATA_URLS], case_sensitive=True),
    cls=NotRequiredIf,
    not_required_if="url",
    help="Download from supported url list instead of URL.",
)
@click.option(
    "--url",
    type=str,
    cls=NotRequiredIf,
    not_required_if="dataset_name",
    help="URL to dataset file.",
)
def download_data(dataset_root, dataset_name, url):
    if dataset_name is not None:
        url = _DOWNLOAD_DATA_URLS[dataset_name]
    download(dataset_root, url)


@omniscient_cli.command()
@click.option(
    "-m",
    "--model-arch",
    type=click.Choice([*_SUPPORTED_MODEL_ARCHS], case_sensitive=True),
    help="Model architecture name.",
    required=True,
)
@click.option(
    "-d",
    "--dataset-name",
    type=str,
    callback=validate_dataset_choice,
    help=f"{'|'.join(_SUPPORTED_DATASETS.keys())}. Supported dataset key or path to dataset.",
)
@click.option(
    "-ds",
    "--data-split",
    callback=validate_data_split,
    cls=PythonLiteralOption,
    type=int,
    help="Train, val, and test split percentages.",
)
@click.option(
    "-f",
    "--subset-fraction",
    type=click.FloatRange(0.0, 1.0),
    default=1.0,
    help="Percentage of dataset to load.",
)
@click.option("-b", "--batch-size", default=8, type=int, help="Batch size for data loader.")
@click.option(
    "-j",
    "--num-workers",
    type=int,
    default=8,
    callback=validate_num_workers,
    help="Number of processes to spin during data loading.",
)
@click.option("--use-gpu/--no-gpu")
@click.option("--gpu-number", type=int, callback=validate_device)
@click.option("--pretrained/--no-pretrained", help="Whether or not to load a model pretrained on ImageNet.")
@click.option(
    "--optimizer", type=click.Choice([*_SUPPORTED_OPTIMIZERS], case_sensitive=True), help="Optimizer class to use."
)
@click.option("-lr", "--learning-rate", type=float, help="Learning rate.")
@click.option("-e", "--num-epochs", type=int, help="Learning rate.")
def train(
    model_arch,
    dataset_name,
    data_split,
    subset_fraction,
    batch_size,
    num_workers,
    use_gpu,
    gpu_number,
    pretrained,
    optimizer,
    learning_rate,
    num_epochs,
):

    dataset_options: DatasetOptions = {
        "dataset_name": dataset_name,
        "subset_fraction": subset_fraction,
        "transformation": "DEFAULT",
    }

    device_options: DeviceOptions = {"use_gpu": use_gpu, "gpu_number": gpu_number}
    trainer_options: TrainerOptions = {
        "model_arch": model_arch,
        "pretrained": pretrained,
        "num_epochs": num_epochs,
    }

    # TODO: add other args
    optimizer_options: OptimizerOptions = {
        "optimizer": _SUPPORTED_OPTIMIZERS[optimizer],
        "learning_rate": learning_rate,
    }

    run(dataset_options, device_options, trainer_options, optimizer_options)


cli = click.CommandCollection(sources=[omniscient_cli, omniscient_datasets])


if __name__ == "__main__":
    cli()
    # omniscient_cli()
