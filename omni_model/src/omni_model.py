import torch.nn as nn
from torchvision import models

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(models.__dict__[name])
    and ("vgg" in name or "resnet" in name)
)


class OmniModel(nn.Module):
    """Model contains a network, two criterions (train, eval) and two metrics."""

    def __init__(
        self,
        model_arch: str,
        num_classes=None,
        pretrained=None,
        device=None,
        # cuda_tf=transforms.ToCuda,
        # detach_tf=transforms.ToDetach,
        # criterions=None,
        # metrics=None, #TODO: add option for is training or not
    ):
        super(OmniModel, self).__init__()
        self.model_arch = model_arch
        if model_arch not in model_names:
            raise ValueError(f"Invalid {model_arch = }. Select from: {model_names}.")
        else:
            self.pretrained = pretrained
            if num_classes is None and self.pretrained is not None:
                self.num_classes = 1000
            elif num_classes is not None:
                self.num_classes = num_classes
            elif num_classes is None and self.pretrained is None:
                raise ValueError(
                    f"Must either load a pretrained model or select the number of outputs for the last layer, i.e. set the num_classes variable."
                )
            temp_model = getattr(models, model_arch)(pretrained=pretrained)
            features = list(temp_model.children())
            in_features = features.pop(-1).in_features
            features.append(nn.Linear(in_features, self.num_classes))
            self.layers = nn.Sequential(*features)
        # self.layers = nn.ModuleList(list(network.modules())[1:])
        # TODO: deal with loading pretrained weights from disk
        # self.criterions = criterions or {}
        # self.metrics = metrics or {}
        if device is None:
            self.device = f"cuda:" + 0 if torch.cuda.is_available() else "cpu"
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

    def cuda(self, device=None):
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