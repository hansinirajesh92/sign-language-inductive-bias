# Sign Language MNIST: CNN vs Vision Transformer Under Data Constraints

This project compares Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) on the Sign Language MNIST dataset under limited-data and full-data training regimes.

## Objective
To study how inductive bias affects model performance when training data is limited.

## Experiments
- CNN with 10% of training data
- CNN with 100% of training data
- ViT with 10% of training data
- ViT with 100% of training data

## Files
- `models/` — model definitions
- `utils/` — data loading, training, evaluation
- `experiments/` — runnable experiment scripts
- `results/` — CSV outputs and generated plots
