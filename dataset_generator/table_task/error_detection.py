import random
import os
import pandas as pd
from dataset_generator.table_serializer import TableSerializer
from dataset_generator.table_task.base_table_task import BaseTableTask


class ErrorDetectionTask(BaseTableTask):
    def __init__(self, sample_size: int = 5):
        """
        初始化错误检测任务
        :param sample_size: 额外随机抽取的行数
        """
        super().__init__()
        self.sample_size = sample_size

    def get_task_descriptions(self, error_type):
        """ 生成任务描述（错误检测），强调 row_id 是主键，error_value 记录原始错误值 """
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
        """ 获取错误检测任务的后缀说明，强调 row_id 来源，error_value 记录检测出的错误值 """
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
        """构造输入表格（脏数据表），格式化为 Markdown"""
        dataset_folder = os.path.dirname(file_path)
        dirty_file = os.path.join(dataset_folder, "dirty.csv")

        if not os.path.exists(dirty_file):
            raise FileNotFoundError(f"dirty.csv 文件未找到: {dirty_file}")

        df = pd.read_csv(dirty_file, dtype=str)
        if "row_id" not in df.columns:
            raise KeyError("dirty.csv 缺少row列，无法索引行号！")

        # 解析 tuple_pairs 选出相关行
        selected_rows = self._extract_tuple_rows(entry, df)
        # 额外随机抽取几行
        additional_rows = self._sample_additional_rows(df, exclude_ids=selected_rows.index)

        # random 生成表格
        final_df = additional_rows.sample(frac=1, random_state=42).reset_index(drop=True)
        # 随机插入 selected_rows
        for _, row in selected_rows.iterrows():
            insert_idx = random.randint(0, len(final_df))  # 生成一个随机插入位置
            final_df = pd.concat(
                [final_df.iloc[:insert_idx], row.to_frame().T, final_df.iloc[insert_idx:]]).reset_index(drop=True)

        # 检查该行是否在final_df中
        # 检查 entry 的 row_id 是否在 final_df 中
        entry_row_id = entry.get("row_id")
        if entry_row_id not in final_df["row_id"].values:
            raise ValueError(f"生成的输入表格中缺少 row_id={entry_row_id}，请检查数据构造逻辑！")

        # 转换为 Markdown 格式
        return TableSerializer.serialize_df(final_df)

    def construct_output(self, entry):
        """ 生成错误检测任务的标注数据 """
        return {
            "row_id": entry["row_id"],
            "column": entry["column"],
            "error_type": entry["error_type"],
            "error_value": entry["error_value"]
        }
