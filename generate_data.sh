#!/bin/bash

# 确保脚本在项目根目录下执行
cd "$(dirname "$0")"

# 指定 Python 解释器（如果使用虚拟环境，可修改 `python` 为 `python3` 或 `python3.8`）
PYTHON=python

# 确保 dataset_generator 目录可被 Python 识别
export PYTHONPATH=.

# 运行数据加载脚本，执行三个任务
$PYTHON dataset_generator/data_loader.py Error_Detection
$PYTHON dataset_generator/data_loader.py Error_Generation
$PYTHON dataset_generator/data_loader.py Error_Correction

echo "✅ Data generation completed!"