#!/bin/bash

# Configuration
HOME="/home/soumya"
SCRIPT_PATH="$HOME/final_ppai/main.py"

# Function to run training for a dataset
run_training() {
    local dataset_path=$1

    python "$SCRIPT_PATH" --data_path "$dataset_path"
    
    if [ $? -eq 0 ]; then
        echo "Successfully completed training for $dataset_path"
    else
        echo "Error processing $dataset_path"
    fi
}

# List of datasets to process
datasets=(
    "$HOME/.medmnist/breastmnist.npz"
    "$HOME/.medmnist/retinamnist.npz"
    "$HOME/.medmnist/pneumoniamnist.npz"
    #! "$HOME/.medmnist/dermamnist.npz" # There is a data leak here so skip
    "$HOME/.medmnist/bloodmnist.npz"
    #! "$HOME/.medmnist/chestmnist.npz" # skipping cos multilabel
    "$HOME/.medmnist/organcmnist.npz"
    "$HOME/.medmnist/organsmnist.npz"
    "$HOME/.medmnist/organamnist.npz"
    "$HOME/.medmnist/pathmnist.npz"
    "$HOME/.medmnist/octmnist.npz"
    "$HOME/.medmnist/tissuemnist.npz"
)

# Run training for all datasets
for dataset in "${datasets[@]}"; do
    run_training "$dataset"
done

echo -e "\nAll experiments completed!"