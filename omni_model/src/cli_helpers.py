import click
import os
import pathlib
from omni_model.src.data.datasets import _SUPPORTED_DATASETS
import torch
import ast


class PythonLiteralOption(click.Option):
    # https://stackoverflow.com/questions/47631914/how-to-pass-several-list-of-arguments-to-click-option
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


def validate_device(ctx, param, value):
    print(dir(ctx))
    print(ctx.params)
    if "use_gpu" not in ctx.params:
        print(
            "WARNING: Option GPU Number cannot be used without setting the `--use-gpu` option. Was this a mistake?"
        )
        print(f"Checking Cuda Available: {(cuda_available:=torch.cuda.is_available())}")
        if cuda_available:
            num_devices = torch.cuda.device_count()
            print(f"Checking device selection: Number of devices found: {num_devices}.")
            if value >= num_devices:
                print(
                    f"WARNING: Max number of devices {num_devices}, but selected {param.name}={value}."
                )
                value = 0
            print(f"Cuda found. Setting device number to {value}.")
        else:
            raise click.BadParameter(
                f"Invalid selection: {param.name} = {value = }. Cuda not detected. Please use the flag `--no-gpu`."
            )
    return value


def validate_dataset_choice(ctx, param, value) -> str:

    if value not in _SUPPORTED_DATASETS.keys():
        potential_path = pathlib.Path(value)
        if potential_path.exists():
            return value
        else:
            raise click.BadParameter(
                f"Invalid selection for {param.name = } {value = }. Use a valid path to a dataset or select from: ({'|'.join(_SUPPORTED_DATASETS)})"
            )
    else:
        return value


def validate_data_split(ctx, param, value):

    if sum(value) != 100:
        raise click.BadParameter(
            f"Invalid {param.name = } {value = }. Data split percentages must sum to 100%."
        )
    else:
        return value
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
        value = num_cpus
    return value
