import torch
import time


class BaseTrainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        data_loaders,
        log_interval=None,
        lr_scheduler=None,
        num_epochs=None,
    ):
        self.model = model
        self.data_loaders = data_loaders
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.criterion = criterion

    def tune(self):
        # TODO: logging + metrics etc
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            self.train_epoch()
            self.evaluate_epoch()

            # TODO: if model is good, save checkpoint
            # TODO: logging stuff

        # TODO: if testing, run evaluate

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

    def adjust_learning_rate(self):
        """Decay the learning rate."""
        pass

    def create_optimizer(self):
        """Initialize the optimizer."""
        pass

    def get_loss(self):
        pass