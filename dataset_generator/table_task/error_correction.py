import random
import os
import pandas as pd
from dataset_generator.table_serializer import TableSerializer
from dataset_generator.table_task.base_table_task import BaseTableTask


class ErrorCorrectionTask(BaseTableTask):
    def __init__(self, sample_size: int = 5):
        """
        初始化错误修复任务
        :param sample_size: 额外随机抽取的行数
        """
        super().__init__()
        self.sample_size = sample_size

    def get_task_descriptions(self, error_type):
        """ 生成任务描述（错误修复），强调 row_id 是主键，修复值存储在 right_value """
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
        """ 获取错误修复任务的后缀说明，强调 row_id 来源，error_value 记录原始错误，right_value 记录修复值 """
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
        """ 生成错误修复任务的标注数据 """
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
