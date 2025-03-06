import random
import os
import pandas as pd
from pathlib import Path
from prompt_builder.table_serializer import TableSerializer
from prompt_builder.table_task.base_table_task import BaseTableTask

class ErrorCorrectionTask(BaseTableTask):
    """Task for correcting errors in a dataset."""


    def __init__(self, sample_size: int = 5):
        """
        Initializes the error correction task.

        Args:
            sample_size (int): Number of additional randomly selected rows.
        """
        super().__init__()
        self.sample_size = sample_size

    def get_task_descriptions(self, error_type):
        """
        Generates task descriptions emphasizing that 'row_id' is the primary key
        and the correct value is stored in 'right_value'.

        Args:
            error_type (str): Type of error to be corrected.

        Returns:
            str: A randomly selected task description.
        """
        descriptions = [
            "Identify and correct a {error_type} error in the row specified by 'row_id'. The 'row_id' uniquely identifies each record and must be taken directly from the input table without modification.",
            "Find a {error_type} error in a specific column of the row identified by 'row_id' and replace it with the correct value. The 'row_id' must remain unchanged as it serves as the primary key.",
            "Detect and fix a {error_type} error by determining the correct value for the erroneous entry in the row specified by 'row_id'. Do NOT modify or generate new 'row_id' values.",
            "Locate an incorrect value in a column of the row identified by 'row_id' and provide the correct replacement. The 'row_id' must be directly referenced from the input table.",
            "Review the table, identify a {error_type} error in a specific row, and replace the erroneous value with the correct one. The 'row_id' must not be altered.",
            "Analyze the input table, find a {error_type} error in a row determined by 'row_id', and correct it by generating the appropriate value.",
            "Detect an incorrect value in the table that qualifies as a {error_type} error and replace it with an accurate entry while preserving 'row_id' from the input table.",
            "Find a {error_type} error in a given row, using 'row_id' as a fixed reference, and provide the correct value while ensuring 'row_id' remains unchanged.",
            "Search the table for a {error_type} error and generate the correct value to replace it. The erroneous value must be recorded, and the correction stored in 'right_value'.",
            "Locate a {error_type} error in the row identified by 'row_id' and return its correct value. Ensure that 'row_id' remains unchanged from the input table."
        ]
        return random.choice(descriptions).format(error_type=error_type)


    def get_suffix(self):
        """
        Provides suffix instructions emphasizing that 'row_id' comes from the input table,
        'error_value' records the original error, and 'right_value' stores the correction.

        Returns:
            str: A randomly selected suffix description.
        """
        descriptions = [
            "Return the final result as a JSON object with 'row_id', 'column', 'error_type', 'error_value', 'right_value', 'missing_value', 'constraint', and 'tuple_pairs'. The 'row_id' must be selected from the input table, 'column' must specify the affected field, 'error_value' must store the detected incorrect value, and 'right_value' must contain the correct replacement.",
            "Provide the output as a JSON object containing 'row_id', 'column', 'error_type', 'error_value', 'right_value', 'missing_value', 'constraint', and 'tuple_pairs'. The 'row_id' must be taken directly from the input table, 'error_value' should reflect the erroneous entry, and 'right_value' should store the corrected value.",
            "Generate a JSON response with 'row_id', 'column', 'error_type', 'error_value', and 'right_value'. The 'row_id' must remain unchanged from the input table, the incorrect value must be recorded in 'error_value', and the correction must be stored in 'right_value'. Only return cell with high confidence.",
            "Output a structured JSON object including 'row_id', 'column', 'error_type', 'error_value', and 'right_value'. Ensure that 'row_id' comes from the input table, 'error_value' reflects the detected erroneous data, and 'right_value' holds the correct value. Only return cell with high confidence.",
            "Return a JSON object with 'row_id', 'column', 'error_type', 'error_value', and 'right_value'. The 'row_id' must strictly match an existing record in the input table, the incorrect value should be stored in 'error_value', and the correct replacement should be provided in 'right_value'.",
            "Provide a JSON output containing 'row_id', 'column', 'error_type', 'error_value', and 'right_value'. Ensure that 'row_id' is taken from the input table, 'error_value' represents the original incorrect value, and 'right_value' stores the correct replacement. Share only cell with high certainty.",
            "Generate a JSON object formatted with 'row_id', 'column', 'error_type', 'error_value', and 'right_value'. The 'row_id' must be directly selected from the input table, 'error_value' should reflect the detected incorrect data, and 'right_value' must store the corrected value. Share only cell with high certainty.",
            "Construct a JSON object with 'row_id', 'column', 'error_type', 'error_value', and 'right_value'. Ensure that 'row_id' is extracted from the input table, 'error_value' holds the incorrect data, and 'right_value' contains the correct value. Only return cell with high confidence."
        ]
        return random.choice(descriptions)

    def construct_input(self, entry, file_path):
        """
        Constructs the input table (dirty data table) and formats it as Markdown.

        Args:
            entry (dict): Error entry containing details of the erroneous data.
            file_path (str): Path to the dataset file.

        Returns:
            str: Markdown-formatted table.

        Raises:
            FileNotFoundError: If 'dirty.csv' is not found in the dataset folder.
            KeyError: If 'row_id' is missing from the dataset.
            ValueError: If the constructed table lacks the required 'row_id'.
        """
        dataset_folder = Path(file_path).parent
        dirty_file = dataset_folder / "dirty.csv"

        if not dirty_file.exists():
            raise FileNotFoundError(f"File not found: {dirty_file}")

        df = pd.read_csv(dirty_file, dtype=str)
        if "row_id" not in df.columns:
            raise KeyError("Missing 'row_id' column in dirty.csv, unable to index rows.")

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
        Generates the labeled output data for the error correction task.

        Args:
            entry (dict): Error entry containing details of the erroneous data.

        Returns:
            dict: JSON-formatted dictionary containing error correction details.
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
