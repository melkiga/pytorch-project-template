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
    data_split: Tuple[int, int, int]
    num_workers: int
    batch_size: int


class ModelOptions:
    def __init__(self):
        pass


class OptimizerOptions:
    def __init__(self):
        pass


class SchedulerOptions:
    def __init__(self):
        pass


class TrainerOptions:
    def __init__(self):
        pass