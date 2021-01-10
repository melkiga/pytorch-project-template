import time
import torch
from omni_model.src.trainer.base_trainer import BaseTrainer


class OmniTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        optimizer=None,
        criterion=None,
        data_loaders=None,
        log_interval=None,
        lr_scheduler=None,
        num_epochs=None,
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
        # predict
        outputs = self.model(images)

        # Loss function
        losses = self.get_loss(outputs, labels)
        loss = losses.item()

        # update gradients
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def get_loss(self):
        if torch.cuda.is_available():
            self.criterion.cuda()
        return self.criterion(pred, label)

    def adjust_learning_rate(self):
        pass

    def evaluate_epoch(self):
        pass

    def evaluate_step(self):
        pass