import torch
import torch.nn as nn


class MNISTModel(nn.Module):
    """MNIST model."""

    def __init__(self) -> None:
        """Performs inheritance and defines the model blocks."""

        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), bias=False),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=32),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), bias=False),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.SiLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=1600, out_features=32),
            nn.SiLU(),
            nn.Linear(in_features=32, out_features=10),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        out = self.block1(data)
        out = self.block2(out)
        out = self.mlp(out.flatten(1))

        return out
