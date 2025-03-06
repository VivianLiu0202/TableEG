import os
import json
import csv
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def repair_dirty_data(dirty_file: str, annotation_file: str, output_clean_file: str):
    """
    Cleans the dirty dataset by applying corrections from an annotation JSONL file.
    If the leftmost column is not `row`, an index column `row` is added, starting from `0`.

    Args:
        dirty_file (str): Path to the dirty dataset (CSV format).
        annotation_file (str): Path to the annotation JSONL file containing corrections.
        output_clean_file (str): Path to save the cleaned dataset (CSV format).

    Raises:
        FileNotFoundError: If the dirty file or annotation file is missing.
        KeyError: If the correction column does not exist in the dataset.
    """
    dirty_file, annotation_file, output_clean_file = map(Path, [dirty_file, annotation_file, output_clean_file])

    if not dirty_file.exists():
        raise FileNotFoundError(f"Dirty file not found: {dirty_file}")
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    df_dirty = pd.read_csv(dirty_file, dtype=str, encoding="utf-8-sig")

    if df_dirty.columns[0] != "row":
        print("Adding 'row' column as index...")
        df_dirty.insert(0, "row", range(len(df_dirty)))

    corrections = []
    with open(annotation_file, "r", encoding="utf-8") as f:
        corrections = [json.loads(line) for line in f]

    for entry in tqdm(corrections, desc="Applying corrections", unit="entry"):
        try:
            row_id = int(entry["row_id"])
            column = entry["column"]
            right_value = entry["right_value"]

            if column in df_dirty.columns:
                df_dirty.at[row_id, column] = right_value
            else:
                print(f"Warning: Column '{column}' not found in {dirty_file}, skipping...")

        except (KeyError, ValueError) as e:
            print(f"Error processing correction {entry}: {e}")

    df_dirty.to_csv(output_clean_file, index=False, encoding="utf-8-sig")
    print(f"Cleaned dataset saved to: {output_clean_file}")


def csv_to_jsonl(csv_file: str, jsonl_file: str):
    """
    Converts a CSV file to JSONL format, removing BOM (ZWNBSP) if present.

    Args:
        csv_file (str): Path to the CSV file.
        jsonl_file (str): Path to save the converted JSONL file.

    Raises:
        FileNotFoundError: If the CSV file is missing.
    """
    csv_file, jsonl_file = Path(csv_file), Path(jsonl_file)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    with open(csv_file, "r", encoding="utf-8-sig") as f, open(jsonl_file, "w", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.lstrip("\ufeff"): v for k, v in row.items()}
            json.dump(row, out, ensure_ascii=False)
            out.write("\n")


def batch_convert_csv_to_jsonl(source_dir: str):
    """
    Recursively scans the source directory and converts all *_annotation.csv files to *_annotation.jsonl.

    Args:
        source_dir (str): Path to the source directory.
    """
    source_dir = Path(source_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith("_annotation.csv"):
                csv_file = Path(root) / file
                jsonl_file = csv_file.with_suffix(".jsonl")

                print(f"Converting: {csv_file} → {jsonl_file}")
                csv_to_jsonl(csv_file, jsonl_file)

    print("All CSV files successfully converted to JSONL.")


def index_row_column(csv_file: str, output_file: str):
    """
    Ensures that the dataset contains a 'row' column. If the first column is 'row',
    it reassigns values starting from 0. Otherwise, it inserts a new 'row' column.

    Args:
        csv_file (str): Path to the input CSV file.
        output_file (str): Path to save the modified CSV file.

    Raises:
        FileNotFoundError: If the input CSV file is missing.
    """
    csv_file, output_file = Path(csv_file), Path(output_file)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file, dtype=str, encoding="utf-8-sig")

    if df.columns[0] == "row":
        print("First column is 'row', reassigning values from 0...")
        df["row"] = range(len(df))
    else:
        print("First column is not 'row', inserting 'row' column...")
        df.insert(0, "row", range(len(df)))

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Dataset modified, 'row' column updated, and saved to {output_file}.")


if __name__ == "__main__":
    # Example: Repair dirty dataset
    repair_dirty_data(
        dirty_file="source/Company/dirty.csv",
        annotation_file="source/University/University_annotation.jsonl",
        output_clean_file="source/University/clean.csv"
    )

    # Example: Convert CSV to JSONL
    batch_convert_csv_to_jsonl("source")

    # Example: Index row column
    index_row_column(
        csv_file="source/Company/dirty.csv",
        output_file="source/Company/dirty.csv"
    )