import torch
from torch import nn
from torchvision.models import resnet18

def build_model(num_classes: int = 100) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return model

def choose_device(requested: str) -> torch.device:
    if requested == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable.")
        return torch.device("mps")
    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
    if requested == "cpu":
        return torch.device("cpu")
    raise RuntimeError("Device must either be mps or cpu")