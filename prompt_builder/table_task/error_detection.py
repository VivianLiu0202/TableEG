import random
import os
import pandas as pd
from pathlib import Path
from prompt_builder.table_serializer import TableSerializer
from prompt_builder.table_task.base_table_task import BaseTableTask

class ErrorDetectionTask(BaseTableTask):
    """Task for detecting errors in a dataset."""

    def __init__(self, sample_size: int = 5):
        """
        Initializes the error detection task.

        Args:
            sample_size (int): Number of additional randomly selected rows.
        """
        super().__init__()
        self.sample_size = sample_size


    def get_task_descriptions(self, error_type):
        """
        Generates task descriptions emphasizing that 'row_id' is the primary key
        and 'error_value' records the detected erroneous value.

        Args:
            error_type (str): Type of error to be detected.

        Returns:
            str: A randomly selected task description.
        """
        descriptions = [
            "Examine the input table and identify a cell in the row specified by 'row_id' that contains a {error_type} error. The 'row_id' uniquely identifies each record and must be taken directly from the input table without modification.",
            "Analyze the input table to find a {error_type} error in a specific column of a row identified by 'row_id'. The 'row_id' must remain unchanged and must be directly referenced from the input table.",
            "Detect a {error_type} error by locating a single erroneous cell in the row specified by 'row_id'. Ensure 'row_id' is preserved as the primary key and remains unaltered.",
            "Review the table and identify an incorrect value in a column of a row determined by 'row_id'. The detected error must be recorded with its exact 'row_id' and column name.",
            "Pinpoint a {error_type} error in the table, ensuring the erroneous cell belongs to a valid 'row_id' from the input table. Record the incorrect value in 'error_value'.",
            "Inspect the table and find a {error_type} error in a specific column of a given 'row_id'. Do NOT modify or generate new 'row_id' values.",
            "Search the input table for a {error_type} error and log its exact position. Ensure that 'row_id' is taken from the input table and is not altered.",
            "Identify a single incorrect data entry that qualifies as a {error_type} error, ensuring that 'row_id' is selected from the input table and remains unchanged.",
            "Detect an erroneous cell in the table that represents a {error_type} issue. The detected row must be referenced using its 'row_id', and the original incorrect value should be stored in 'error_value'.",
            "Find a {error_type} error in the table and document the corresponding 'row_id', column, and incorrect value while ensuring 'row_id' is directly taken from the input table."
        ]
        return random.choice(descriptions).format(error_type=error_type)

    def get_suffix(self):
        """
        Provides suffix instructions emphasizing that 'row_id' comes from the input table,
        and 'error_value' records the detected erroneous value.

        Returns:
            str: A randomly selected suffix description.
        """
        descriptions = [
            "Return the final result as a JSON object with 'row_id', 'column', 'error_type', and 'error_value'. The 'row_id' must be taken from the input table without modification, 'column' must specify the affected field, and 'error_value' must store the detected erroneous value.",
            "Provide the output as a JSON object containing 'row_id', 'column', 'error_type', and 'error_value'. Ensure that 'row_id' is directly referenced from the input table and 'error_value' contains the original incorrect data. Only include confident cell.",
            "Generate a JSON response with 'row_id', 'column', 'error_type', and 'error_value'. The 'row_id' should come from the input table, and 'error_value' must capture the detected incorrect value without modification. Only include erroneous cell with high confidence.",
            "Output a structured JSON object including 'row_id', 'column', 'error_type', and 'error_value'. The 'row_id' must be extracted from the input table, and 'error_value' should store the exact erroneous value from the detected cell. Share only cell with high certainty.",
            "Return a JSON object with 'row_id', 'column', 'error_type', and 'error_value'. The 'row_id' must strictly match an existing record in the input table, and 'error_value' should contain the original incorrect value found in that row. Share only cell with high certainty.",
            "Provide a JSON output containing 'row_id', 'column', 'error_type', and 'error_value'. Ensure that 'row_id' remains unchanged from the input table and 'error_value' reflects the erroneous entry as detected. Report only cell you are highly confident about.",
            "Generate a JSON result formatted with 'row_id', 'column', 'error_type', and 'error_value'. The 'row_id' must be directly selected from the input table, and 'error_value' must store the incorrect data exactly as found. Only include confident cell.",
            "Construct a JSON object with 'row_id', 'column', 'error_type', and 'error_value'. Ensure that 'row_id' is taken from the input table, and 'error_value' correctly reflects the detected erroneous data entry. Only return cell with high confidence."
        ]
        return random.choice(descriptions)

    def construct_input(self, entry, file_path):
        """
        Constructs the input table (dirty data table) and formats it as Markdown.

        Args:
            entry (dict): Error entry specifying the detected error.
            file_path (str): Path to the dataset file.

        Returns:
            str: Markdown-formatted table.

        Raises:
            FileNotFoundError: If 'dirty.csv' is missing.
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
        Generates the labeled output data for the error detection task.

        Args:
            entry (dict): Error entry specifying the detected error.

        Returns:
            dict: JSON-formatted dictionary containing detected error details.
        """
        return {
            "row_id": entry["row_id"],
            "column": entry["column"],
            "error_type": entry["error_type"],
            "error_value": entry["error_value"]
        }
