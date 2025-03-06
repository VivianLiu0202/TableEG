import os
import sys
import argparse
import numpy as np
from pathlib import Path
from prompt_builder.data_generator import DataGenerator
from table_task.table_task_factory import TableTaskFactory
import random

# Set project root dynamically
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
def set_random_seed(seed=2):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def main(task_name: str):
    """Main function to generate dataset for table error tasks."""
    source_dir = os.path.join(BASE_DIR, "source")

    # Retrieve the corresponding task processing class
    table_task = TableTaskFactory.get_table_task(task_name)
    data_generator = DataGenerator(table_task, source_dir=source_dir, verbose=True)

    # Split ratio for training and testing sets
    split_ratio = 0.9
    data_generator.generate_data(split_ratio)

    print("Dataset has been successfully generated and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset for table error tasks")
    parser.add_argument(
        "task_name",
        type=str,
        choices=["Error_Generation", "Error_Detection", "Error_Correction"],
        help="Task type for dataset generation"
    )

    args = parser.parse_args()
    main(args.task_name)