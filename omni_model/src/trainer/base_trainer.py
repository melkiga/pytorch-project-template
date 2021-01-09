import torch


class BaseTrainer:
    def __init__(
        self,
        net,
        optimizer,
        criterion,
        data_loaders,
        log_interval=None,
        lr_scheduler=None,
        modelname=None,
    ):
        pass

    def train_step(self):
        """Implement the training logic for a step."""
        raise NotImplementedError

    def train_epoch(self):
        """Implement the training logic for a single epoch.
        This includes calculating and recording losses etc.
        """
        raise NotImplementedError

    def evaluate_epoch(self):
        """Implement the validation logic for a single epoch."""
        pass

    def evaluate_step(self):
        """Implement the validation logic for a single step."""
        pass

    def tune(self):
        pass

    def adjust_learning_rate(self):
        """Decay the learning rate."""
        pass

    def create_optimizer(self):
        """Initialize the optimizer."""
        pass

    def get_loss(self):
        pass