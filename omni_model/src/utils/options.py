from typing import TypedDict, Tuple, Union, Callable, List, Any
import json
from torchvision import transforms

# TODO: fill this with base option as base class
class BaseOptions(TypedDict):
    placeholder: None


class TransformOptions(TypedDict):
    transform: Union[str, None, Callable[[List[Any]], transforms.Compose]]


class DatasetOptions(TypedDict):
    dataset_name: str
    subset_fraction: float


class CIFARDatasetOptions(DatasetOptions, total=False):
    transformation: Any
    is_training: bool


class DataLoaderOptions(TypedDict):
    data_split: Tuple[int, int, int]
    num_workers: int
    batch_size: int


class DeviceOptions(TypedDict, total=False):
    use_gpu: bool
    gpu_number: int


class ModelOptions(TypedDict):
    model_arch: str


class AllModelOptions(ModelOptions, total=False):
    num_classes: int
    pretrained: Union[str, bool]


class OptimizerOptions(TypedDict):
    optimizer_name: str
    learning_rate: float


class SchedulerOptions(TypedDict):
    scheduler_name: str


class TrainerOptions(AllModelOptions, total=False):
    num_epochs: int
