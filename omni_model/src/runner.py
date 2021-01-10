# TODO: https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
from omni_model.src.utils.options import DatasetOptions
from omni_model.src.data.datasets import CIFAR10Dataset


def run(dataset_options: DatasetOptions = None):
    dataset = CIFAR10Dataset(**dataset_options)

    # TODO: load parameters

    # TODO: set device

    # TODO: initialize engine
    # engine can train, eval, optimize the model
    # engine can save and load the model and optimizer

    # TODO: load dataset
    # dataset is a dictionary that contains all the needed datasets indexed by modes
    # (example: dataset.keys() -> ['train','eval'])

    # TODO: load model
    # model includes a network, a criterion and a metric
    # model can register engine hooks (begin epoch, end batch, end batch, etc.)
    # (example: "calculate mAP at the end of the evaluation epoch")
    # note: model can access to datasets using engine.dataset

    # setup training params
    # optimizer can register engine hooks

    # setup logger

    # setup metrics

    # TODO: set up a visualizer for pretty plots

    # setup output directory