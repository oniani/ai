import torch
import torch.nn as nn


VGG_VARIANT: dict[int, list[int | str]] = {
    11: [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    13: [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    16: [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    # 19: [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    """A generic VGG model representation."""

    def __init__(self, variant: int) -> None:
        """Performs inheritance and defines model blocks."""

        super().__init__()

        self.features = self.make_layers(VGG_VARIANT[variant])
        self.classifier = nn.Linear(512, 10)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass."""

        out = self.features(data)
        out = out.flatten(1)
        out = self.classifier(out)

        return out

    def make_layers(self, model_config: list[int | str]) -> nn.Sequential:
        """Uses the configuration and builds up a VGG model."""

        layers = []
        in_channels = 3
        for val in model_config:
            if isinstance(val, int):
                layer = [
                    nn.Conv2d(in_channels=in_channels, out_channels=val, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=val),
                    nn.ReLU(),
                ]
                layers.extend(layer)
                in_channels = val
            elif val == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                raise ValueError(f"Unknown value: {val}")

        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))

        return nn.Sequential(*layers)
