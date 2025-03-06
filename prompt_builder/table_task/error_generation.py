import random
import os
import pandas as pd
from pathlib import Path
from prompt_builder.table_serializer import TableSerializer
from prompt_builder.table_task.base_table_task import BaseTableTask

class ErrorGenerationTask(BaseTableTask):
    """Task for generating artificial errors in a dataset."""

    def __init__(self, sample_size: int = 5):
        """
        Initializes the error generation task.

        Args:
            sample_size (int): Number of additional randomly selected rows.
        """
        super().__init__()
        self.sample_size = sample_size

    def get_task_descriptions(self, error_type):
        """
        Generates task descriptions emphasizing that 'row_id' is the primary key
        and errors occur in specific columns.

        Args:
            error_type (str): Type of error to be introduced.

        Returns:
            str: A randomly selected task description.
        """
        descriptions = [
            "Modify a single attribute in the row identified by 'row_id' to introduce a {error_type} error. Ensure that the 'row_id' column is preserved as it uniquely identifies each record. Do NOT modify or generate new 'row_id' values.",
            "Introduce an error in a specific column of the row identified by 'row_id', creating a {error_type} error. The 'row_id' column must remain unchanged as it serves as the primary key, uniquely identifying each row, even if the table order is shuffled.",
            "Alter one column of a given row (selected using 'row_id') to create a {error_type} error. The row to be modified must be referenced using its 'row_id', which uniquely identifies each entry, regardless of the table order.",
            "Corrupt a single data entry in the row specified by 'row_id' to simulate a {error_type} error. Ensure 'row_id' is directly taken from the input table and remains unaltered.",
            "Modify a specific attribute in the row identified by 'row_id' to introduce a {error_type} error. Do NOT change or create new 'row_id' values.",
            "Change one column in the row determined by 'row_id' to generate a {error_type} error. The 'row_id' must be taken from the input table and remain unchanged.",
            "Introduce an error into a specific cell of the row identified by 'row_id', resulting in a {error_type} error. The 'row_id' serves as a fixed identifier and should not be modified."
        ]
        return random.choice(descriptions).format(error_type=error_type)

    def get_suffix(self):
        """
        Provides suffix instructions emphasizing that 'row_id' comes from the input table,
        'error_value' records the generated error, and 'right_value' stores the original value.

        Returns:
            str: A randomly selected suffix description.
        """
        descriptions = [
            "Provide the final result as a JSON object with 'row_id', 'column', 'error_type', 'error_value', 'right_value', 'missing_value', 'constraint', and 'tuple_pairs'. Select 'row_id' from the input table, modify only a single column, store the correct value in 'right_value', and the erroneous value in 'error_value'. Only include erroneous cell with high confidence.",
            "Output a JSON object containing 'row_id', 'column', 'error_type', 'error_value', 'right_value', 'missing_value', 'constraint', and 'tuple_pairs'. The 'row_id' must be from the input table, and the modified column’s correct value should be recorded in 'right_value', with the introduced error in 'error_value'. Report only error you are highly confident about.",
            "Generate a JSON result with 'row_id', 'column', 'error_type', 'error_value', 'right_value', 'missing_value', 'constraint', and 'tuple_pairs'. Ensure that 'row_id' is taken from the input table, the original correct value is placed in 'right_value', and the incorrect generated value is stored in 'error_value'. Include only high-confidence erroneous cell.",
            "Return a JSON object structured with 'row_id', 'column', 'error_type', 'error_value', 'right_value', 'missing_value', 'constraint', and 'tuple_pairs'. The 'row_id' must not be altered, and the original value should be saved in 'right_value' while the generated incorrect value should be placed in 'error_value'. Report only error with high certainty.",
            "Provide a JSON output including 'row_id', 'column', 'error_type', 'error_value', 'right_value', 'missing_value', 'constraint', and 'tuple_pairs'. The modified column's original data must be stored in 'right_value', and the generated incorrect entry should be recorded in 'error_value'. Only return error with high confidence.",
            "Output a structured JSON object with 'row_id', 'column', 'error_type', 'error_value', 'right_value', 'missing_value', 'constraint', and 'tuple_pairs'. The 'row_id' must come from the input table, and only a single column should be modified. Keep the correct value in 'right_value' and the incorrect generated value in 'error_value'. Only include confident error.",
            "Generate a JSON object containing 'row_id', 'column', 'error_type', 'error_value', 'right_value', 'missing_value', 'constraint', and 'tuple_pairs'. Ensure that 'row_id' is extracted from the input table, the modified column’s original value is recorded in 'right_value', and the erroneous value is stored in 'error_value'. Share only error with high certainty."
        ]
        return random.choice(descriptions)

    def construct_input(self, entry, file_path):
        """
        Constructs the input table by modifying a clean dataset and formatting it as Markdown.

        Args:
            entry (dict): Error entry specifying the modification.
            file_path (str): Path to the dataset file.

        Returns:
            str: Markdown-formatted table.

        Raises:
            FileNotFoundError: If 'clean.csv' is missing.
            KeyError: If 'row_id' is missing from the dataset.
            ValueError: If the constructed table lacks the required 'row_id'.
        """
        dataset_folder = Path(file_path).parent
        clean_file = dataset_folder / "clean.csv"

        if not clean_file.exists():
            raise FileNotFoundError(f"File not found: {clean_file}")

        df = pd.read_csv(clean_file, dtype=str)
        if "row_id" not in df.columns:
            raise KeyError("Missing 'row_id' column in clean.csv, unable to index rows.")

        # Extract relevant rows based on 'tuple_pairs'
        selected_rows = self._extract_tuple_rows(entry, df)

        # Randomly sample additional rows
        additional_rows = self._sample_additional_rows(df, exclude_ids=selected_rows.index)

        # Shuffle table
        final_df = additional_rows.sample(frac=1, random_state=42).reset_index(drop=True)

        # Insert selected rows at random positions
        for _, row in selected_rows.iterrows():
            insert_idx = random.randint(0, len(final_df))
            final_df = pd.concat(
                [final_df.iloc[:insert_idx], row.to_frame().T, final_df.iloc[insert_idx:]]
            ).reset_index(drop=True)

        # Validate that 'row_id' exists in final_df
        entry_row_id = entry.get("row_id")
        if entry_row_id not in final_df["row_id"].values:
            raise ValueError(f"Generated input table lacks row_id={entry_row_id}, check data construction logic.")

        # Convert to Markdown format
        return TableSerializer.serialize_df(final_df)

    def construct_output(self, entry):
        """
        Generates the labeled output data for the error generation task.

        Args:
            entry (dict): Error entry specifying the generated error.

        Returns:
            dict: JSON-formatted dictionary containing error details.
        """
        return {
            "row_id": entry["row_id"],
            "column": entry["column"],
            "error_type": entry["error_type"],
            "error_value": entry["error_value"],
            "right_value": entry["right_value"],
            "missing_value": entry.get("missing_value", 0),
            "constraint": entry.get("constraint", ""),
            "tuple_pairs": entry.get("tuple_pairs", "")
        }
