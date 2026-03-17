from experiments.runncnn import run_experiment

if __name__ == "__main__":
    run_experiment(
        subset_fraction=0.1,
        epochs=15,
        weight_decay=0.0,
        results_file="results/cnn_10.csv"
    )