import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_transforms(config):
    """Build the preparation steps applied to every image."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config["data"]["mean"], config["data"]["std"]),
    ])


def get_dataloaders(config):
    """Download MNIST, split it, and return train, validation, and test DataLoaders."""
    data_cfg = config["data"]
    transform = get_transforms(config)

    # Load the full training set (60,000 images) and the test set (10,000 images)
    full_train = datasets.MNIST(root=data_cfg["root"], train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_cfg["root"], train=False, download=True, transform=transform)

    # Split the full training set into training and validation sets
    val_size = data_cfg["val_size"]
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(config["seed"])
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    # Settings shared by all three DataLoaders
    num_workers = data_cfg["num_workers"]
    loader_settings = {
        "batch_size": data_cfg["batch_size"],
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(train_set, shuffle=True, **loader_settings)
    val_loader = DataLoader(val_set, shuffle=False, **loader_settings)
    test_loader = DataLoader(test_set, shuffle=False, **loader_settings)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # This test only runs when the file is run directly, not when it is imported.
    import matplotlib.pyplot as plt

    config = load_config()
    train_loader, val_loader, test_loader = get_dataloaders(config)

    print(f"Training images:   {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Test images:       {len(test_loader.dataset)}")

    images, labels = next(iter(train_loader))
    print(f"Batch of images shape: {tuple(images.shape)}")
    print(f"Batch of labels shape: {tuple(labels.shape)}")

    plt.imshow(images[0].squeeze(), cmap="gray")
    plt.title(f"Label: {labels[0].item()}")
    plt.axis("off")
    plt.show()