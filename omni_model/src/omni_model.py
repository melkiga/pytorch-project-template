from typing import Optional
import torch
import torch.nn as nn
from torchvision import models

model_names = models.list_models(module=models)


class OmniModel(nn.Module):
    """Model contains a network, two criterions (train, eval) and two metrics."""

    def __init__(
        self,
        model_arch: str,
        num_classes: int = None,
        weights: bool = None,
        device: int = None,
        # cuda_tf=transforms.ToCuda,
        # detach_tf=transforms.ToDetach,
        # criterions=None,
        # metrics=None, #TODO: add option for is training or not (feature extracting) https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    ) -> None:
        super(OmniModel, self).__init__()

        self.model_arch = model_arch
        if model_arch not in model_names:
            raise ValueError(f"Invalid {model_arch = }. Select from: {model_names}.")
        else:
            # TODO: deal with loading pretrained weights from disk
            self.weights = None
            if weights:
                weights_enum = models.get_model_weights(model_arch)
                self.weights = weights_enum.DEFAULT

            temp_model = getattr(models, model_arch)(weights=self.weights)
            features = list(temp_model.children())

            if num_classes is None and self.weights is not None:
                self.num_classes = len(self.weights.meta["categories"])
            elif num_classes is not None:
                self.num_classes = num_classes
                in_features = features.pop(-1).in_features
                features.append(nn.Linear(in_features, self.num_classes))
            elif num_classes is None and self.weights is None:
                raise ValueError(
                    f"Must either load a pretrained model or select the number of outputs for the last layer, i.e. set the num_classes variable."
                )

            self.layers = nn.Sequential(*features)

        # self.layers = nn.ModuleList(list(network.modules())[1:])

        # self.criterions = criterions or {}
        # self.metrics = metrics or {}

        if device is None:
            self.device = f"cuda:{0}" if torch.cuda.is_available() else "cpu"
        self.is_cuda = False

        # self.eval()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def eval(self):
        """Activate evaluation mode"""
        super(OmniModel, self).train(mode=False)
        self.mode = "eval"

    def train(self):
        """Activate training mode"""
        super(OmniModel, self).train(mode=True)
        self.mode = "train"

    def freeze_layers(self):
        self.eval()
        for layer in self.parameters():
            layer.requires_grad = False

    def cuda(self, device: Optional[int] = None):
        """Moves all model parameters and buffers to the GPU.
        Args:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.is_cuda = True
        self.device = device
        return self._apply(lambda t: t.cuda(device))

    def cpu(self):
        """Moves all model parameters and buffers to the CPU."""
        self.is_cuda = False
        return self._apply(lambda t: t.cpu())
