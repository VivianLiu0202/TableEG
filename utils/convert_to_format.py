import os
import csv
import json

def csv_to_jsonl(csv_file, jsonl_file):
    """ 读取 CSV 并转换为 JSONL，去除 BOM (ZWNBSP) """
    with open(csv_file, 'r', encoding='utf-8-sig') as f, open(jsonl_file, 'w', encoding='utf-8') as out:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.lstrip("\ufeff"): v for k, v in row.items()}  # 去除可能的 BOM
            json.dump(row, out, ensure_ascii=False)
            out.write("\n")

def batch_convert_csv_to_jsonl(source_dir):
    """ 递归扫描 source 目录，转换所有 *_annotation.csv 为 *_annotation.jsonl """
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith("_annotation.csv"):
                csv_file = os.path.join(root, file)
                jsonl_file = os.path.join(root, file.replace(".csv", ".jsonl"))

                print(f"📂 Converting: {csv_file} → {jsonl_file}")
                csv_to_jsonl(csv_file, jsonl_file)

    print("✅ 所有 CSV 文件已成功转换为 JSONL！")


if __name__ == "__main__":
    source_dir = "source"
    batch_convert_csv_to_jsonl(source_dir)