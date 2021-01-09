import click
import os
import pathlib
from omni_model.src.datasets.data import _SUPPORTED_DATASETS


def validate_dataset_choice(ctx, param, value):

    if value not in _SUPPORTED_DATASETS.keys():
        potential_path = pathlib.Path(value)
        if potential_path.exists():
            return (param, value)
        else:
            raise click.BadParameter(
                f"Invalid selection for {param.name = } {value = }. Use a valid path to a dataset or select from: ({'|'.join(_SUPPORTED_DATASETS)})"
            )
    else:
        return (param, value)


def validate_data_split(ctx, param, value):

    if sum(value) != 100:
        raise click.BadParameter(
            f"Invalid {param.name = } {value = }. Data split percentages must sum to 100%."
        )
    # TODO: more checks? what if we had: 99,1,0


def validate_num_workers(ctx, param, value):
    num_cpus = os.cpu_count()
    if value > num_cpus:
        click.echo(
            click.style(
                f"ATTENTION - Worker number {value = } is larget than number of CPUs {num_cpus}. Capping workers to CPU number.",
                fg="yellow",
                bold=True,
            )
        )
