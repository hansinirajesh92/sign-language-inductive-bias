from experiments.runncnn import run_experiment

if __name__ == "__main__":
    run_experiment(
        subset_fraction=1.0,
        epochs=15,
        weight_decay=1e-4,
        results_file="results/cnn_100.csv"
    )