from typing import TypedDict, Tuple
import json


class BaseOptions(TypedDict):
    def __init__(self):
        pass


class DatasetOptions(TypedDict):
    dataset_name: str
    subset_fraction: float
    data_split: Tuple[int, int, int]
    num_workers: int
    batch_size: int

    def __str__(self):
        return json.dumps(self.__dict__(), indent=4, sort_keys=True)
        # return {**self.__dict__()}


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