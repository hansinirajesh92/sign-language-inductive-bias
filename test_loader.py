from utils.data_loader import get_data_loaders

train_loader, test_loader = get_data_loaders(
    train_csv_path="data/sign_mnist_train.csv",
    test_csv_path="data/sign_mnist_test.csv",
    batch_size=64,
    subset_fraction=0.1
)

images, labels = next(iter(train_loader))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Unique labels in batch:", labels[:10])