import os
import pandas as pd
import matplotlib.pyplot as plt

cnn10 = pd.read_csv("results/cnn_10.csv")
cnn100 = pd.read_csv("results/cnn_100.csv")
vit10 = pd.read_csv("results/vit_10.csv")
vit100 = pd.read_csv("results/vit_100.csv")

os.makedirs("results/plots", exist_ok=True)

plt.figure(figsize=(9, 6))
plt.plot(cnn10["epoch"], cnn10["test_acc"], label="CNN 10%")
plt.plot(cnn100["epoch"], cnn100["test_acc"], label="CNN 100%")
plt.plot(vit10["epoch"], vit10["test_acc"], label="ViT 10%")
plt.plot(vit100["epoch"], vit100["test_acc"], label="ViT 100%")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("results/plots/test_accuracy_vs_epoch.png")
plt.close()

plt.figure(figsize=(9, 6))
plt.plot(cnn10["epoch"], cnn10["train_acc"], label="CNN 10%")
plt.plot(cnn100["epoch"], cnn100["train_acc"], label="CNN 100%")
plt.plot(vit10["epoch"], vit10["train_acc"], label="ViT 10%")
plt.plot(vit100["epoch"], vit100["train_acc"], label="ViT 100%")
plt.xlabel("Epoch")
plt.ylabel("Train Accuracy")
plt.title("Train Accuracy vs Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("results/plots/train_accuracy_vs_epoch.png")
plt.close()

plt.figure(figsize=(9, 6))
plt.plot(cnn10["epoch"], cnn10["test_loss"], label="CNN 10%")
plt.plot(cnn100["epoch"], cnn100["test_loss"], label="CNN 100%")
plt.plot(vit10["epoch"], vit10["test_loss"], label="ViT 10%")
plt.plot(vit100["epoch"], vit100["test_loss"], label="ViT 100%")
plt.xlabel("Epoch")
plt.ylabel("Test Loss")
plt.title("Test Loss vs Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("results/plots/test_loss_vs_epoch.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(cnn10["epoch"], cnn10["train_acc"], label="Train Accuracy")
plt.plot(cnn10["epoch"], cnn10["test_acc"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN 10%: Train vs Test Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("results/plots/cnn10_train_vs_test.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(cnn100["epoch"], cnn100["train_acc"], label="Train Accuracy")
plt.plot(cnn100["epoch"], cnn100["test_acc"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN 100%: Train vs Test Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("results/plots/cnn100_train_vs_test.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(vit10["epoch"], vit10["train_acc"], label="Train Accuracy")
plt.plot(vit10["epoch"], vit10["test_acc"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ViT 10%: Train vs Test Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("results/plots/vit10_train_vs_test.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(vit100["epoch"], vit100["train_acc"], label="Train Accuracy")
plt.plot(vit100["epoch"], vit100["test_acc"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ViT 100%: Train vs Test Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("results/plots/vit100_train_vs_test.png")
plt.close()

final_test_acc = {
    "CNN 10%": cnn10["test_acc"].iloc[-1],
    "CNN 100%": cnn100["test_acc"].iloc[-1],
    "ViT 10%": vit10["test_acc"].iloc[-1],
    "ViT 100%": vit100["test_acc"].iloc[-1],
}

plt.figure(figsize=(8, 5))
plt.bar(final_test_acc.keys(), final_test_acc.values())
plt.ylabel("Final Test Accuracy")
plt.title("Final Test Accuracy by Model and Data Fraction")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/plots/final_test_accuracy_bar.png")
plt.close()

best_test_acc = {
    "CNN 10%": cnn10["test_acc"].max(),
    "CNN 100%": cnn100["test_acc"].max(),
    "ViT 10%": vit10["test_acc"].max(),
    "ViT 100%": vit100["test_acc"].max(),
}

plt.figure(figsize=(8, 5))
plt.bar(best_test_acc.keys(), best_test_acc.values())
plt.ylabel("Best Test Accuracy")
plt.title("Best Test Accuracy by Model and Data Fraction")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/plots/best_test_accuracy_bar.png")
plt.close()

summary_df = pd.DataFrame({
    "Model": ["CNN", "CNN", "ViT", "ViT"],
    "Data Fraction": [0.1, 1.0, 0.1, 1.0],
    "Best Test Accuracy": [
        cnn10["test_acc"].max(),
        cnn100["test_acc"].max(),
        vit10["test_acc"].max(),
        vit100["test_acc"].max(),
    ]
})

cnn_summary = summary_df[summary_df["Model"] == "CNN"]
vit_summary = summary_df[summary_df["Model"] == "ViT"]

plt.figure(figsize=(8, 5))
plt.plot(cnn_summary["Data Fraction"], cnn_summary["Best Test Accuracy"], marker="o", label="CNN")
plt.plot(vit_summary["Data Fraction"], vit_summary["Best Test Accuracy"], marker="o", label="ViT")
plt.xlabel("Data Fraction")
plt.ylabel("Best Test Accuracy")
plt.title("Best Test Accuracy vs Data Fraction")
plt.legend()
plt.tight_layout()
plt.savefig("results/plots/accuracy_vs_data_fraction.png")
plt.close()

print("All plots saved in results/plots/")
print(summary_df)