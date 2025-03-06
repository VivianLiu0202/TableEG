import os
import sys
import copy
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Union
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from table_task.base_table_task import BaseTableTask
from table_task.table_task_factory import TableTaskFactory

# Define project root dynamically
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

def split_data(data, train_ratio):
    """Splits data into training and testing sets based on a given ratio."""
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

class DataGenerator:
    def __init__(self,
                 table_task: Union[str, BaseTableTask],
                 source_dir: str = str(BASE_DIR / "source"),
                 sample_size: int = 5,
                 n_jobs: int = 1,
                 verbose: bool = True):
        """Initializes the data generator with task-specific settings."""
        if isinstance(table_task, str):
            self.table_task = TableTaskFactory.get_table_task(table_task, sample_size=sample_size)
        else:
            self.table_task = table_task
        self.source_dir = source_dir
        self.n_jobs = n_jobs
        self.verbose = verbose

    def generate_data(self, split_ratio: float = 0.9):
        """Scans the source directory for annotation files and generates training data."""
        self.print_log("Scanning annotation files in source directory...")
        # annotation_files= ['/Users/liuvivian/table_tuning_for_error_generating_task/source/Company/Company_annotation.jsonl']
        # annotation_files = ['/Users/liuvivian/table_tuning_for_error_generating_task/source/hospital/hospital_annotation.jsonl']
        # annotation_files = ['/Users/liuvivian/table_tuning_for_error_generating_task/source/rayyan/rayyan_annotation.jsonl']
        annotation_files = []
        if not os.path.exists(self.source_dir) or not os.path.isdir(self.source_dir):
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith("_annotation.jsonl"):
                    annotation_files.append(os.path.join(root, file))

        if not annotation_files:
            raise ValueError("No annotation files found in the source directory.")

        all_data = []
        for file in tqdm(annotation_files, desc="Processing annotation files", unit="file"):
            all_data.extend(self.process_annotation_file(file))

        # Split data into training and testing sets
        train_data, test_data_zeroshot = split_data(all_data, split_ratio)

        # Deep copy for few-shot testing data
        test_data_fewshot = copy.deepcopy(test_data_zeroshot)

        # Build error_type-to-training data index mapping
        error_type_dict = defaultdict(list)
        for item in train_data:
            error_type_dict[item["output"]["error_type"]].append(item)

        # Generate few-shot samples for test data
        for i in range(len(test_data_fewshot)):
            error_type = test_data_fewshot[i]["output"]["error_type"]
            if error_type in error_type_dict and len(error_type_dict[error_type]) > 1:
                num_example = min(2, len(error_type_dict[error_type]))
                example_idxs = random.sample(error_type_dict[error_type], num_example)
                for example in example_idxs:
                    test_data_fewshot[i]['instruction'] += f"\nInput:\n{example['input']}\nOutput:\n{example['output']}\n"

        # Generate few-shot samples for training data
        for i, item in enumerate(train_data):
            error_type = item["output"]["error_type"]
            if error_type in error_type_dict and len(error_type_dict[error_type]) > 1:
                num_example = random.randint(0, min(2, len(error_type_dict[error_type]) - 1))
                example_idxs = random.sample([ex for ex in error_type_dict[error_type] if ex != item], num_example)
                for example in example_idxs:
                    item['instruction'] += f"\nInput:\n{example['input']}\nOutput:\n{example['output']}\n"

        # Save all processed datasets
        self.save_dataset(train_data, test_data_zeroshot, test_data_fewshot)


    def process_annotation_file(self, file_path: str):
        """Parses a single `_annotation.jsonl` file and generates structured data."""
        self.print_log(f"Processing file: {file_path}")

        max_samples_per_type = 2500  # 每种 error_type 的最大数量
        error_type_dict = defaultdict(list)  # 记录每种 error_type 的数据

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()  # 读取所有行，获取文件总行数

        # Categorize data by error type
        for line in lines:
            try:
                entry = json.loads(line)
                error_type = entry["error_type"]
                error_type_dict[error_type].append(line)
            except json.JSONDecodeError as e:
                self.print_log(f"JSON parsing error in {file_path}: {e}")
                continue

        # Limit the number of samples per error type
        filtered_lines = []
        for error_type, records in error_type_dict.items():
            selected_records = records[:max_samples_per_type] if len(records) > max_samples_per_type else records
            filtered_lines.extend(selected_records)

        results = []
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            for data in tqdm(
                    executor.map(lambda eachline: self.safe_parse_json(eachline, file_path), filtered_lines),
                    total=len(filtered_lines), desc=f"Processing {os.path.basename(file_path)}", unit="entry"
            ):
                if data:
                    results.append(data)

        return results


    def safe_parse_json(self, line, file_path):
        """Safely parses JSON to prevent crashes due to corrupt data."""
        try:
            entry = json.loads(line)
            return self.generate_data_entry(entry, file_path)
        except json.JSONDecodeError as e:
            self.print_log(f"JSON parsing error in {file_path}: {e}")
            return None

    def generate_data_entry(self, entry, file_path):
        """Generates an (instruction, input, output) data structure."""
        error_type = entry["error_type"]
        instruction = self.table_task.generate_instruction(entry)
        input_table = self.table_task.construct_input(entry, file_path)
        output_json = self.table_task.construct_output(entry)
        return {
            "instruction": instruction,
            "input": input_table,
            "output": output_json
        }

    def save_dataset(self, train_data, test_data_zeroshot, test_data_fewshot):
        """Saves the training and test datasets in JSONL format."""
        dataset_dir = BASE_DIR / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        train_dir = dataset_dir / "train"
        test_dir = dataset_dir / "test"
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)

        task_name = self.table_task.__class__.__name__.replace("Task", "")

        train_file = train_dir / f"train_{task_name}.jsonl"
        test_file_zeroshot = test_dir / f"test_{task_name}_zeroshot.jsonl"
        test_file_fewshot = test_dir / f"test_{task_name}_fewshot.jsonl"

        self.write_jsonl(train_file, train_data)
        self.write_jsonl(test_file_zeroshot, test_data_zeroshot)
        self.write_jsonl(test_file_fewshot, test_data_fewshot)

        print(f"Dataset generation completed: {task_name}")

    @staticmethod
    def write_jsonl(filepath, data):
        """Writes a list of JSON objects to a JSONL file."""
        with open(filepath, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def print_log(self, *args):
        """ 打印日志 """
        if self.verbose:
            print(*args)
