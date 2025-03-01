import os
import json
import pandas as pd
from tqdm import tqdm


def repair_dirty_data(dirty_file: str, annotation_file: str, output_clean_file: str):
    """
    修复 dirty 数据集，根据 annotation.jsonl 纠正错误，生成 clean 数据集。
    如果最左侧列不是 `row`，则添加 `row` 作为索引列，并从 `0` 开始编号。

    :param dirty_file: 脏数据文件路径（CSV 格式）
    :param annotation_file: 标注错误的 JSONL 文件路径
    :param output_clean_file: 生成的干净数据集（CSV）
    """

    # 1️⃣ 读取 dirty 数据集
    df_dirty = pd.read_csv(dirty_file)

    # 2️⃣ 检查是否有 `row` 列，若没有，则添加 `row` 编号
    if df_dirty.columns[0] != "row":
        print("🚀 Adding 'row' column as index...")
        df_dirty.insert(0, "row", range(len(df_dirty)))  # 添加从 0 开始的行号

    # 3️⃣ 读取 annotation.jsonl 并存入字典
    corrections = []
    with open(annotation_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            corrections.append(entry)

    # 4️⃣ 逐行修复数据
    for entry in tqdm(corrections, desc="Repairing dataset_old", unit="entry"):
        try:
            row_id = int(entry["row_id"])  # 行号
            column = entry["column"]  # 错误的列
            right_value = entry["right_value"]  # 正确值

            if column in df_dirty.columns:
                df_dirty.at[row_id, column] = right_value  # 修复错误值
            else:
                print(f"⚠️ Warning: Column {column} not found in {dirty_file}, skipping...")

        except Exception as e:
            print(f"❌ Error repairing row {entry}: {e}")

    # 5️⃣ 保存修复后的数据集
    df_dirty.to_csv(output_clean_file, index=False)
    print(f"✅ 修复完成，clean 数据集已保存至: {output_clean_file}")


# 示例调用
repair_dirty_data(
    dirty_file="source/Company/dirty.csv",
    annotation_file="source/University/University_annotation.jsonl",
    output_clean_file="source/University/clean.csv"
)