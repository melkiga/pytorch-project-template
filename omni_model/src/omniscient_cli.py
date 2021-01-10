import click
from omni_model.src.cli_helpers import (
    validate_dataset_choice,
    validate_data_split,
    validate_num_workers,
    PythonLiteralOption,
)
from omni_model.src.model.omni_model import model_names as _SUPPORTED_MODEL_ARCHS
from omni_model.src.data.dataset_helpers import _SUPPORTED_DATASETS
from omni_model.src.utils.options import DatasetOptions
from omni_model.src.runner import run


@click.group()
def omniscient_cli():
    pass


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
    type=click.IntRange(0, 1),
    help="Percentage of dataset to load.",
)
@click.option("-b", "--batch-size", type=int, help="Batch size for data loader.")
@click.option(
    "-j",
    "--num-workers",
    type=int,
    callback=validate_num_workers,
    help="Number of processes to spin during data loading.",
)
def omni_training_cli(
    model_arch, dataset_name, data_split, subset_fraction, batch_size, num_workers
):

    dataset_options: DatasetOptions = {
        "dataset_name": dataset_name,
        "data_split": data_split,
        "subset_fraction": subset_fraction,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    run(dataset_options)


if __name__ == "__main__":
    omniscient_cli()