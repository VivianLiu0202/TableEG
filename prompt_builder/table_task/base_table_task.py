import os
import random
import pandas as pd
import re
from pathlib import Path
from prompt_builder.table_serializer import TableSerializer

# Define project root dynamically
BASE_DIR = Path(__file__).resolve().parent.parent

class BaseTableTask:
    """Base class for all table-related tasks."""

    def __init__(self):
        pass

    def get_task_descriptions(self, error_type):
        """Retrieve task descriptions based on error type."""
        raise NotImplementedError  # Implement in subclass

    def get_error_type_descriptions(self, error_type):
        """Retrieve descriptions for specific error types.

        Args:
            error_type (str): Type of error.

        Returns:
            str: A randomly selected description for the given error type.
        """
        """ 获取错误类型的描述 """
        descriptions = {
            "rule_violation": [
                "Rule violations occur when data entries fail to satisfy predefined integrity constraints, such as uniqueness, referential integrity, or functional dependencies. Populate the 'tuple_pairs' field with violating tuples and the 'constraint' field with the broken rule.",
                "A rule violation happens when data contradicts database constraints, leading to inconsistencies and reducing overall data reliability. The 'tuple_pairs' field should contain the conflicting tuples, and 'constraint' should specify the violated rule.",
                "Rule violations include duplicate primary keys, conflicting attribute values, and invalid foreign key references within a dataset_old. Fill 'tuple_pairs' with related erroneous tuples and 'constraint' with the specific constraint that was violated.",
                "A rule violation represents a logical inconsistency where data fails to meet structural constraints, potentially causing errors in downstream applications. Ensure 'tuple_pairs' lists the affected tuples and 'constraint' details the violated integrity rule."
            ],
            "pattern_violation": [
                "Pattern violations refer to data values that do not conform to expected syntax, structure, or semantic constraints.",
                "A pattern violation occurs when a data entry deviates from its expected format, such as incorrect date formats or improperly structured email addresses.",
                "Pattern violations include formatting errors, misplaced values, and syntax inconsistencies that disrupt data processing and interpretation.",
                "A pattern violation is an incorrect representation of data that fails to match predefined format rules, affecting readability and usability."
            ],
            "outliers": [
                "Outliers are data points that significantly deviate from the expected distribution within a column, either numerically or categorically.",
                "Outliers represent extreme or unexpected values in a dataset_old that do not conform to the general pattern of the data.",
                "Outliers are anomalies in numerical or categorical data that can distort statistical analysis and impact decision-making.",
                "Outliers occur when data values fall outside the typical range, often due to measurement errors or data entry mistakes."
            ],
            "missing_value": [
                "Missing values refer to data cells that are empty, incorrectly filled, or implicitly missing due to format conversion issues or inconsistent data integration.",
                "A missing value is an absent or null entry in a dataset_old, which can lead to incomplete analysis and unreliable predictions.",
                "Missing values occur when essential data is unavailable, either due to collection errors, manual omissions, or placeholder values like 'N/A'.",
                "Missing data represents gaps in a dataset_old where values are expected but not provided, potentially introducing bias in analysis."
            ]
        }
        return random.choice(descriptions.get(error_type, ["Unknown error type."]))

    def get_suffix(self):
        """Retrieve suffix for error type.

        Returns:
            str: Suffix related to the specific error type.
        """
        raise NotImplementedError  # Implement in subclass

    def generate_instruction(self, entry):
        """Generate a complete instruction consisting of task description and error description.

        Args:
            entry (dict): Error entry containing relevant details.

        Returns:
            str: Formulated instruction based on the error entry.
        """
        error_type = entry["error_type"]
        constraint = entry.get("constraint", "")
        task_desc = self.get_task_descriptions(error_type)
        error_desc = self.get_error_type_descriptions(error_type)
        suffix_desc = self.get_suffix()
        if constraint:
            constraint_desc = self.extract_constraint(constraint)
            return f"{task_desc} {error_desc} {constraint_desc} {suffix_desc}"
        else:
            return f"{task_desc} {error_desc} {suffix_desc}"

    def construct_input(self, entry, file_path):
        """Construct the input table based on entry details."""
        raise NotImplementedError

    def construct_output(self, entry):
        """Construct the expected output."""
        pass

    def _extract_tuple_rows(self, entry, df):
        """Extract corresponding rows based on tuple pairs.

        Args:
            entry (dict): Error entry containing `tuple_pairs`.
            df (pd.DataFrame): DataFrame containing the dataset.

        Returns:
            pd.DataFrame: Extracted rows based on the error type.
        """
        error_type = entry.get("error_type", "")
        row_id = entry.get("row_id", "")

        if error_type == "rule_violation":
            tuple_pairs = entry.get("tuple_pairs", "")
            row_ids = [int(x.strip()) for x in tuple_pairs.strip("()").split(",") if x.strip().isdigit()]
            if len(row_ids) < 2:
                return pd.DataFrame()  # Return empty if insufficient rows
            selected_rows = df[df["row_id"].astype(int).isin(row_ids[:2])]  # Select only the first two rows
        else:
            if not row_id.isdigit():
                return pd.DataFrame()  # Return empty if row_id is invalid
            selected_rows = df[df["row_id"].astype(int) == int(row_id)]

        return selected_rows

    def _sample_additional_rows(self, df, exclude_ids):
        """Randomly sample additional rows while excluding specified ones.

        Args:
            df (pd.DataFrame): Dataset as a DataFrame.
            exclude_ids (list): List of row IDs to be excluded.

        Returns:
            pd.DataFrame: Sampled additional rows.
        """
        available_rows = df[~df["row_id"].astype(int).isin(exclude_ids)]
        return available_rows.sample(n=min(self.sample_size, len(available_rows)), random_state=42)

    def extract_constraint(self, constraint):
        """Parse and transform constraint descriptions.

        Args:
            constraint (str): Constraint description.

        Returns:
            str: Transformed constraint description with entity mapping.
        """
        constraint = constraint.replace("Constraint Violation: ", "")
        row_numbers = re.findall(r'Row (\d+)', constraint)
        if len(row_numbers) < 2:
            return "Invalid constraint format."

        first_row, second_row = row_numbers[:2]
        entity_map = {first_row: "entity 1", second_row: "entity 2"}

        for row, entity in entity_map.items():
            constraint = constraint.replace(f"Row {row}", entity)

        return "Denial Constraint is: " + constraint