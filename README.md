<h1 align="center">Towards Practical Benchmarking of Data Cleaning Techniques: On Generating Authentic Errors via Large Language Models</h1>
A framework for generating realistic tabular data errors using LLMs

# Overview

### Abstract

Data quality remains an important challenge in data-driven systems, as errors in tabular data can severely compromise downstream analytics and machine learning performance. Although numerous error detection algorithms have been proposed, the lack of diverse, real-world error datasets limits comprehensive evaluation. Manual error annotation is both time-consuming and inconsistent, motivating the exploration of synthetic error generation as an alternative. In this work, we introduce **TableEG**, a framework that leverages large language models (LLMs) to generate authentic errors. By employing a table fine-tuning strategy and a triplet representation \((I, T, O)\) to model error generation, detection, and correction tasks, TableEG captures the complex dependencies inherent in two-dimensional tables. Trained on 12 real-world datasets spanning 10 diverse domains, TableEG ensures that the synthesized errors faithfully reflect authentic error distributions. Experimental results indicate that errors generated by TableEG exhibit superior pattern and distribution similarity compared to both rule-based methods and LLM-generated errors without fine-tuning. Furthermore, performance metrics on TableEG-generated errors closely align with those on real-world errors across nearly all datasets and detection algorithms, particularly for machine learning based detection techniques. Overall, TableEG not only bridges the gap between synthetic and real-world errors but also establishes a robust benchmark for subsequent error detection and correction tasks.

### Framework Overview

![framework_00](./readme_img/framework_00.png)

# Usage

### Environment Setup
```bash
git clone git@github.com:viviancircle/TableEG.git
cd TableEG
# We use Python3.8, PyTorch2.4.1.
conda env create -f TableEG_env.yaml
conda activate TableEG_env
cd TableEG
```

### Dataset Preparation
| Dataset            | #Rows  | #Columns | Domain         | Error Rate | Error Types (M/P/R/O) |
|--------------------|--------|----------|---------------|------------|----------------------|
| Rayyan            | 1,000  | 11       | Academic      | 8.62%      | M, P, R |
| Company           | 128,889| 7        | Business      | 34.21%     | P, R |
| Marketing         | 8,993  | 14       | Business      | 21.29%     | M, P |
| Movie (Metadata)  | 7,390  | 17       | Entertainment | 6.10%      | M, P |
| Movie (Box Office)| 9,329  | 7        | Entertainment | 7.31%      | P |
| Credit            | 150,000| 10       | Finance       | 2.33%      | M, O |
| Beers             | 2,410  | 11       | Food          | 12.66%     | M, P, R |
| Restaurant        | 12,007 | 10       | Food          | 0.53%      | P |
| Hospital         | 1,000  | 20       | Health        | 2.55%      | P, R |
| Airbnb            | 42,492 | 40       | Hospitality   | 0.22%      | M, O |
| University        | 286    | 17       | Education     | 13.97%     | P |
| Sensor           | 62,076 | 8        | Technology    | 0.01%      | O |
| Flights          | 2,376  | 7        | Transportation| 24.15%     | M, P, R |


### Prompt_Builder
```bash
bash generate_data.sh
```

### Trainer
```bash
python train/train_llama3_lora.py
```

### Error_generator
```bash
python error_generator/generate_error.py
```

### Evaluator
```bash
python evaluation/Error_Distribution_Evaluation.py
python evaluation/Error_Pattern_Evaluation.py
```





