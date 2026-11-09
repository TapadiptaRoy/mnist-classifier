from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from src.data import get_dataloaders, load_config
from src.model import build_model
from src.train import evaluate
from src.utils import get_device, set_seed


def load_trained_model(config, device):
    model = build_model(config).to(device)
    checkpoint_path = config["training"]["checkpoint_path"]
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def get_predictions(model, loader, device):
    """Run the model on every image and collect all predictions and true labels."""
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    return torch.cat(all_preds), torch.cat(all_labels)


def build_confusion_matrix(preds, labels, num_classes):
    """Count how many images of each true digit were predicted as each digit."""
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for true, pred in zip(labels.tolist(), preds.tolist()):
        matrix[true, pred] += 1
    return matrix


def save_confusion_matrix_plot(matrix, path):
    """Draw the confusion matrix as a colored grid and save it as an image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    num_classes = matrix.shape[0]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(matrix.numpy(), cmap="Blues")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title("Confusion matrix (test set)")

    # Write each count inside its cell
    threshold = matrix.max().item() / 2
    for i in range(num_classes):
        for j in range(num_classes):
            value = matrix[i, j].item()
            color = "white" if value > threshold else "black"
            ax.text(j, i, value, ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Load the best model, test it, and report the results."""
    config = load_config()
    set_seed(config["seed"])
    device = get_device()
    print(f"Using device: {device}")

    _, _, test_loader = get_dataloaders(config)
    model = load_trained_model(config, device)

    # Overall test loss and accuracy
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"\nTest loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.2%}")

    # Confusion matrix and accuracy for each digit
    preds, labels = get_predictions(model, test_loader, device)
    num_classes = config["model"]["num_classes"]
    matrix = build_confusion_matrix(preds, labels, num_classes)

    print("\nAccuracy per digit:")
    for digit in range(num_classes):
        correct = matrix[digit, digit].item()
        total = matrix[digit].sum().item()
        print(f"  {digit}: {correct / total:.2%}  ({correct}/{total})")

    plot_path = config["evaluation"]["confusion_matrix_path"]
    save_confusion_matrix_plot(matrix, plot_path)
    print(f"\nConfusion matrix saved to: {plot_path}")


if __name__ == "__main__":
    main()