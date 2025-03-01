import pandas as pd

class TableSerializer:
    @staticmethod
    def serialize_df(df: pd.DataFrame, entry=None) -> str:
        """
        将 DataFrame 转换为 Markdown 表格
        :param df: 待序列化的表格数据
        :param entry: 错误标注 entry，可用于修正 missing value
        :return: Markdown 格式的表格字符串
        """
        if df.empty:
            return ""

        headers = df.columns.tolist()
        table_str = "| " + " | ".join(headers) + " |\n"
        table_str += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        # 处理 entry 相关的缺失值
        for _, row in df.iterrows():
            row_values = []
            for col in headers:
                value = row[col]

                if entry and entry["error_type"] == "missing_value":
                    if str(row["row"]) == str(entry["row_id"]) and col == entry["column"]:
                        value = entry["error_value"]  # 使用错误标注的值填充
                    else:
                        value = value if pd.notna(value) else ""  # 其他行仍然按原始数据展示

                row_values.append(str(value))

            table_str += "| " + " | ".join(row_values) + " |\n"

        return table_str.strip()