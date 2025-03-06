#!/bin/bash

# Ensure the script is executed from the project root directory
cd "$(dirname "$0")" || exit 1

# Define the Python interpreter (modify if using a virtual environment)
PYTHON=${PYTHON:-python3}  # Defaults to python3 if PYTHON is not set

# Ensure prompt_builder is recognized by Python
export PYTHONPATH="$(pwd)"

# Run data loading scripts for all tasks
TASKS=("Error_Detection" "Error_Generation" "Error_Correction")

for TASK in "${TASKS[@]}"; do
    echo "Running data generation for: $TASK"
    $PYTHON dataset_generator/data_loader.py "$TASK" || {
        echo "Error occurred while processing $TASK."
        exit 1
    }
done

echo "Data generation completed successfully!"