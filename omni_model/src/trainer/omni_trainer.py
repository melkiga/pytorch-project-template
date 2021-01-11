import time
import torch
from omni_model.src.trainer.base_trainer import BaseTrainer
from omni_model.src.data.datasets import DataLoaderWrapper


class OmniTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss,
        data_loaders: DataLoaderWrapper,
        num_epochs: int,
        log_interval=None,
        lr_scheduler=None,
    ):
        super().__init__(
            model,
            optimizer=optimizer,
            criterion=criterion,
            data_loaders=data_loaders,
            log_interval=log_interval,
            lr_scheduler=lr_scheduler,
            num_epochs=num_epochs,
        )

    def train_epoch(self):
        self.model.train()
        data_loader = self.data_loaders.training_dataloader

        for batch_index, (images, labels) in enumerate(data_loader):
            images = images.to(self.model.device)
            labels = labels.to(self.model.device)

            self.train_step(images, labels)

            # TODO: logging something

    def train_step(self, images, labels):
        # update gradients
        self.optimizer.zero_grad()

        # predict
        outputs = self.model(images)
        _, preds = torch.max(outputs, 1)

        # Loss function
        losses = self.get_loss(outputs, labels)
        loss = losses.item()

        losses.backward()
        self.optimizer.step()

        print(torch.sum(preds == labels.data))

    def get_loss(self, pred, label):
        if self.model.is_cuda:
            self.criterion.cuda()
        return self.criterion(pred, label)

    def adjust_learning_rate(self):
        pass

    def evaluate_epoch(self):
        pass

    def evaluate_step(self):
        pass