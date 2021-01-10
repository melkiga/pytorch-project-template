from typing import TypedDict, Tuple, Union, Callable, List, Any
from torch.optim import SGD, Adam
from torchvision import transforms

_SUPPORTED_OPTIMIZERS = {"SGD": SGD, "ADAM": Adam}
import torch

# TODO: fill this with base option as base class
class BaseOptions(TypedDict):
    placeholder: None


class TransformOptions(TypedDict):
    transform: Union[str, None, Callable[[List[Any]], transforms.Compose]]


class RequiredDatasetOptions(TypedDict):
    dataset_name: str


class DatasetOptions(RequiredDatasetOptions, total=False):
    transformation: Any
    is_training: bool
    subset_fraction: float


class DataLoaderOptions(TypedDict, total=False):
    data_split: Tuple[int, int, int]
    num_workers: int
    batch_size: int


class DeviceOptions(TypedDict, total=False):
    use_gpu: bool
    gpu_number: int


class RequiredModelOptions(TypedDict):
    model_arch: str


class ModelOptions(RequiredModelOptions, total=False):
    num_classes: int
    pretrained: Union[str, bool]


class RequiredOptimizerOptions(TypedDict):
    optimizer: torch.optim.Optimizer
    learning_rate: float


class SGDOptimizerOptions(RequiredOptimizerOptions, total=False):
    momentum: float
    dampening: float
    nesterov: bool


class AdamOptimizerOptions(RequiredOptimizerOptions, total=False):
    betas: Tuple[float, float]
    eps: float
    amsgrad: bool


class OptimizerOptions(SGDOptimizerOptions, AdamOptimizerOptions, total=False):
    weight_decay: float


class SchedulerOptions(TypedDict):
    scheduler_name: str


class TrainerOptions(ModelOptions, OptimizerOptions, SchedulerOptions):
    num_epochs: int
