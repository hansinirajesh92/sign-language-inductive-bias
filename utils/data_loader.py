import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class SignLanguageMNISTDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)

        raw_labels = df.iloc[:, 0].values
        self.images = df.iloc[:, 1:].values.astype("float32") / 255.0
        self.images = self.images.reshape(-1, 1, 28, 28)

        # remap labels to contiguous range: 0, 1, 2, ..., 23
        unique_labels = sorted(set(raw_labels))
        self.label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        self.labels = [self.label_map[label] for label in raw_labels]

        print(f"Loaded {csv_file}")
        print(f"Original labels: {unique_labels}")
        print(f"Label mapping: {self.label_map}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def get_data_loaders(
    train_csv_path,
    test_csv_path,
    batch_size=64,
    subset_fraction=1.0
):
    train_dataset = SignLanguageMNISTDataset(train_csv_path)
    test_dataset = SignLanguageMNISTDataset(test_csv_path)

    if subset_fraction < 1.0:
        subset_size = int(len(train_dataset) * subset_fraction)
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader