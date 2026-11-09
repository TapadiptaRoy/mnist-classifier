import torch
from torch import nn

class DigitCNN(nn.Module):
    """A small CNN: two convolutional blocks followed by two linear layers."""

    def __init__(self, conv1_channels=32, conv2_channels=64,
                 hidden_units=128, dropout=0.25, num_classes=10):
        super().__init__()

        # Part 1: feature extractor (finds patterns in the image)
        self.features = nn.Sequential(
            # Block 1: (1, 28, 28) -> (32, 14, 14)
            nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: (32, 14, 14) -> (64, 7, 7)
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Part 2: classifier (makes the decision)
        self.classifier = nn.Sequential(
            nn.Flatten(),                                       # (64, 7, 7) -> (3136)
            nn.Linear(conv2_channels * 7 * 7, hidden_units),    # (3136) -> (128)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes),               # (128) -> (10)
        )

    def forward(self, x):
        """Describe how a batch of images flows through the network."""
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(config):
    """Create a DigitCNN using the settings from the config file."""
    model_cfg = config["model"]
    return DigitCNN(
        conv1_channels=model_cfg["conv1_channels"],
        conv2_channels=model_cfg["conv2_channels"],
        hidden_units=model_cfg["hidden_units"],
        dropout=model_cfg["dropout"],
        num_classes=model_cfg["num_classes"],
    )


def count_parameters(model):
    """Count how many numbers (weights) the model can learn."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # This test only runs when the file is run directly, not when it is imported.
    from src.data import load_config

    config = load_config()
    model = build_model(config)

    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    # Pass a fake batch through the model to check the shapes
    fake_images = torch.randn(64, 1, 28, 28)
    output = model(fake_images)
    print(f"Input shape:  {tuple(fake_images.shape)}")
    print(f"Output shape: {tuple(output.shape)}")