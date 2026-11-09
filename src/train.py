"""
train.py
Trains the CNN on MNIST, checks it on the validation set after every epoch,
and saves the best model.
"""

from pathlib import Path

import torch
from torch import nn

from src.data import get_dataloaders, load_config
from src.model import build_model
from src.utils import get_device, set_seed


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Train the model on every batch once. Return average loss and accuracy."""
    model.train()
    total_loss, total_correct, total_images = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)              # 1. Predict
        loss = loss_fn(outputs, labels)      # 2. Measure how wrong
        optimizer.zero_grad()                # 3. Clear old gradients
        loss.backward()                      # 4. Calculate new gradients
        optimizer.step()                     # 5. Update the weights

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_images += batch_size

    return total_loss / total_images, total_correct / total_images


def evaluate(model, loader, loss_fn, device):
    """Check the model on a dataset without learning. Return average loss and accuracy."""
    model.eval()
    total_loss, total_correct, total_images = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_images += batch_size

    return total_loss / total_images, total_correct / total_images


def main():
    """Load everything, train for several epochs, and save the best model."""
    config = load_config()
    train_cfg = config["training"]

    set_seed(config["seed"])
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(config)
    model = build_model(config).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    checkpoint_path = Path(train_cfg["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    epochs = train_cfg["epochs"]

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        message = (
            f"Epoch {epoch}/{epochs} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.2%} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.2%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            message += " | saved best model"

        print(message)

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2%}")
    print(f"Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()