import os
import sys
import argparse
import numpy as np

ROOT = "/Users/liuvivian/table_tuning_for_error_generating_task"
sys.path.append("/Users/liuvivian/table_tuning_for_error_generating_task")
from dataset_generator.data_generator import DataGenerator
from table_task.table_task_factory import TableTaskFactory
import random

def random_everything():
    random.seed(2)
    np.random.seed(2)

def main(task_name: str):
    source_dir = os.path.join(ROOT,"source")

    # 获取对应的任务处理类
    table_task = TableTaskFactory.get_table_task(task_name)
    data_generator = DataGenerator(table_task, source_dir=source_dir, verbose=True)

    split_ratio = 0.9 # 训练集和测试集的比例
    data_generator.generate_data(split_ratio)
    print(f"数据集已保存")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset_old for error table tasks")
    parser.add_argument("task_name", type=str, choices=["Error_Generation", "Error_Detection", "Error_Correction"],
                        help="Task type to generate dataset_old")

    args = parser.parse_args()
    # main(args.task_name, args.test_file, args.fewshot_test)
    main(args.task_name)
    # main("Error_Detection")
