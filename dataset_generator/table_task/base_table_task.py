import random
import os
import pandas as pd
import re
from dataset_generator.table_serializer import TableSerializer
class BaseTableTask:
    """ 基础表格任务类，所有任务继承它 """
    def __init__(self):
        pass

    def get_task_descriptions(self, error_type):
        """ 任务描述，基于任务类型 (error_generation, error_detection, error_correction) """
        raise NotImplementedError  # 交给子类实现

    def get_error_type_descriptions(self, error_type):
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
        """ 获取错误类型的后缀 """
        raise NotImplementedError  # 交给子类实现

    def generate_instruction(self, entry):
        """ 生成完整的 instruction，由任务描述 + 错误类型描述 组成 """
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
        raise NotImplementedError

    def construct_output(self, entry):
        pass

    def _extract_tuple_rows(self, entry, df):
        """ 根据 tuple_pairs 解析出对应行 """
        error_type = entry.get("error_type", "")
        row_id = entry.get("row_id", "")

        if error_type == "rule_violation":
            # 需要 至少 选择两行
            tuple_pairs = entry.get("tuple_pairs", "")
            row_ids = [int(x.strip()) for x in tuple_pairs.strip("()").split(",") if x.strip().isdigit()]
            if len(row_ids) < 2:
                print(f"rule_violation 错误，但 tuple_pairs 不足 2 行: {row_ids}")
                return pd.DataFrame()  # 若不足 2 行，则返回空 DataFrame
            selected_rows = df[df["row_id"].astype(int).isin(row_ids[:2])]  # 仅取前 2 行
        else:
            if not row_id.isdigit():
                print(f"非 rule_violation，但 row_id 不是数字: {row_id}")
                return pd.DataFrame()  # row_id 无效，返回空
            selected_rows = df[df["row_id"].astype(int) == int(row_id)]

        return selected_rows

    def _sample_additional_rows(self, df, exclude_ids):
        """
        额外随机抽取 sample_size 行，排除已选的行
        :param df: 数据表 (DataFrame)
        :param exclude_ids: 需要排除的行的 row_id 列表
        :return: 采样后的 DataFrame
        """
        # 先过滤掉 exclude_ids 中的行
        available_rows = df[~df["row_id"].astype(int).isin(exclude_ids)]

        # 随机采样 `self.sample_size` 行（如果可选行数不足，则取最大可能值）
        return available_rows.sample(n=min(self.sample_size, len(available_rows)), random_state=42)

    def extract_constraint(self, constraint):
        """
        解析 constraint 字符串，将所有 Row xxx 替换为 entity 1/entity 2，并直接返回修改后的内容。
        """

        # 去掉开头的 'Constraint Violation: '
        constraint = constraint.replace("Constraint Violation: ", "")
        
        # 提取所有涉及的行号，并映射为 entity 1 和 entity 2
        row_numbers = re.findall(r'Row (\d+)', constraint)
        if len(row_numbers) < 2:
            return "Invalid constraint format."

        first_row, second_row = row_numbers[:2]  # 仅取前两个 row_id
        entity_map = {first_row: "entity 1", second_row: "entity 2"}

        # 替换所有 Row XXX 为 entity 1/entity 2
        for row, entity in entity_map.items():
            constraint = constraint.replace(f"Row {row}", entity)

        return "Denial Constraint is: " + constraint

    # def extract_constraint(self, constraint):
    #     """
    #     解析 constraint 字符串，转换为自然语言描述，并确保所有 Row xxx 只被替换一次，不重复，并处理 and 分隔的多个约束。
    #     """
    #     # 提取所有涉及的行号，并映射为 entity 1 和 entity 2
    #     row_numbers = re.findall(r'Row (\d+)', constraint)
    #     if len(row_numbers) < 2:
    #         return "Invalid constraint format."
    #
    #     first_row, second_row = row_numbers[:2]  # 仅取前两个 row_id
    #     entity_map = {first_row: "entity 1", second_row: "entity 2"}
    #
    #     # 替换所有 Row XXX 为 entity 1/entity 2
    #     for row, entity in entity_map.items():
    #         constraint = constraint.replace(f"Row {row}", entity)
    #
    #     # 按 and 分割多个约束
    #     constraint_parts = constraint.split(", and ")
    #     descriptions = []
    #     seen = set()  # 避免重复的约束句子
    #
    #     for part in constraint_parts:
    #         if "is equal to" in part:
    #             match = re.search(r'The (.*?) of entity 1 is equal to the (.*?) of entity 2', part)
    #             if match:
    #                 desc = f"The {match.group(1)} of entity 1 is equal to the {match.group(2)} of entity 2."
    #                 if desc not in seen:
    #                     descriptions.append(desc)
    #                     seen.add(desc)
    #         elif "is different from" in part:
    #             match = re.search(r'The (.*?) of entity 1 is different from the (.*?) of entity 2', part)
    #             if match:
    #                 desc = f"The {match.group(1)} of entity 1 is different from the {match.group(2)} of entity 2."
    #                 if desc not in seen:
    #                     descriptions.append(desc)
    #                     seen.add(desc)
    #
    #     return "Denial Constraint is: " + " ".join(descriptions)

