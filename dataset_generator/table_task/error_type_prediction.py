import random
import os
import pandas as pd
from dataset_generator.table_serializer import TableSerializer
from dataset_generator.table_task.base_table_task import BaseTableTask
class ErrorTypePredictionTask(BaseTableTask):
    """
    错误类型预测任务：让模型先判断当前表格更适合什么错误类型
    """
    def get_task_descriptions(self, error_type):
        """ 生成任务描述（错误类型预测） """
        descriptions = [
            "Analyze the given table and determine the most likely type of error that could occur.",
            "Based on the given table, identify whether a missing_value, outliers, pattern_violation, or rule_violation is more probable.",
            "Examine the table and predict which type of error is most likely to appear: missing_value, outliers, pattern_violation, or rule_violation.",
            "Review the dataset and determine the most probable error type based on the table's structure and content."
        ]
        return random.choice(descriptions)

    def get_suffix(self):
        """ 获取输出格式 """
        return "Provide the final result as a JSON object with the field 'error_type'."

    def construct_output(self, entry):
        """ 生成错误类型预测任务的标注数据 """
        return {"error_type": entry["error_type"]}