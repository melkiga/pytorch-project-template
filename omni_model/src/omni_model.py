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
        model_arch=None,
        num_classes=None,
        pretrained=None,
        device=None,
        # cuda_tf=transforms.ToCuda,
        # detach_tf=transforms.ToDetach,
        # criterions=None,
        # metrics=None,
    ):
        super(OmniModel, self).__init__()
        self.model_arch = model_arch
        if model_arch not in model_names:
            raise ValueError(f"Invalid {model_arch = }. Select from: {model_names}.")
        else:
            network = getattr(models, model_arch)(pretrained=pretrained)
        self.layers = nn.ModuleList(network.modules())
        # TODO: deal with loading pretrained weights from disk
        # self.criterions = criterions or {}
        # self.metrics = metrics or {}
        self.is_cuda = False
        # self.eval()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

    def eval(self):
        """Activate evaluation mode"""
        super(OmniModel, self).train(mode=False)
        self.mode = "eval"

    def train(self):
        """Activate training mode"""
        super(OmniModel, self).train(mode=True)
        self.mode = "train"

    def load(self):
        pass

    def cuda(self, device=None):
        """Moves all model parameters and buffers to the GPU.
        Args:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.is_cuda = True
        return self._apply(lambda t: t.cuda(device))

    def cpu(self):
        """Moves all model parameters and buffers to the CPU."""
        self.is_cuda = False
        return self._apply(lambda t: t.cpu())