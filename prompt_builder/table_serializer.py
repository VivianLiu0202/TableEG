import pandas as pd

class TableSerializer:
    @staticmethod
    def serialize_df(df: pd.DataFrame, entry=None) -> str:
        """
        Converts a DataFrame into a Markdown table format.

        Args:
            df (pd.DataFrame): The input DataFrame to be serialized.
            entry (dict, optional): Error annotation entry, used for correcting missing values.

        Returns:
            str: A Markdown-formatted table as a string.
        """
        if df.empty:
            return ""

        headers = df.columns.tolist()
        table_str = "| " + " | ".join(headers) + " |\n"
        table_str += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        # Process missing value corrections based on the provided annotation entry
        for _, row in df.iterrows():
            row_values = []
            for col in headers:
                value = row[col]

                if entry and entry.get("error_type") == "missing_value":
                    if str(row.get("row", "")) == str(entry.get("row_id", "")) and col == entry.get("column"):
                        value = entry.get("error_value", "")  # Fill in the annotated error value
                    else:
                        value = value if pd.notna(value) else ""  # Keep other values unchanged

                row_values.append(str(value))

            table_str += "| " + " | ".join(row_values) + " |\n"

        return table_str.strip()