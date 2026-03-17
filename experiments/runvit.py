import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from models.vit import SimpleViT
from utils.data_loader import get_data_loaders
from utils.train import train_one_epoch, evaluate


def run_experiment(subset_fraction, epochs, weight_decay=0.0, results_file=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"Running ViT with subset_fraction={subset_fraction}, epochs={epochs}, weight_decay={weight_decay}")

    train_loader, test_loader = get_data_loaders(
        train_csv_path="data/sign_mnist_train.csv",
        test_csv_path="data/sign_mnist_test.csv",
        batch_size=64,
        subset_fraction=subset_fraction
    )

    model = SimpleViT(num_classes=24).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    rows = []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )

        rows.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "subset_fraction": subset_fraction,
            "weight_decay": weight_decay,
        })

    if results_file is not None:
        os.makedirs("results", exist_ok=True)
        with open(results_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved results to {results_file}")