import os
import pandas as pd

# 定义数据文件所在的文件夹
source_folder = "/Users/liuvivian/table_tuning_for_error_generating_task/source"

# 遍历 `source_folder` 及其子目录，查找 `clean.csv` 和 `dirty.csv`
for root, dirs, files in os.walk(source_folder):
    for file_name in files:
        if file_name in ["clean.csv", "dirty.csv"]:
            file_path = os.path.join(root, file_name)
            print(f"正在处理: {file_path}")

            try:
                # 读取 CSV 文件
                df = pd.read_csv(file_path, dtype=str, encoding="utf-8-sig")
                if df.columns[0] == "row_id":
                    print(f"{file_name}: 第一列是 'row_id'，跳过处理...")
                    continue

                if df.columns[0] == "row":
                    print(f"{file_name}: 第一列是 'row'，删除该列并插入新的 'row_id' 列...")
                    df.drop(columns=["row"], inplace=True)

                # 在最左侧插入新的 'row_id' 列
                df.insert(0, "row_id", range(len(df)))

                # 保存修改后的文件
                df.to_csv(file_path, index=False, encoding="utf-8-sig")
                print(f"{file_name}: 处理完成，已保存至 {file_path}\n")

            except Exception as e:
                print(f"处理 {file_name} 时出错，跳过该文件: {e}\n")