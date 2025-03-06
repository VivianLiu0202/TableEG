import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
import json
from tqdm import tqdm
import os
import pandas as pd


class ErrorQualityEvaluator:
    """ Evaluates the similarity between LLM-generated errors and real-world errors. """
    def __init__(self, tokenizer, model, annotation_paths, k=1, verbose=True):
        """
        Initialize the quality evaluator.

        Args:
            tokenizer (AutoTokenizer): The tokenizer for LLM.
            model (AutoModelForCausalLM): The LLM model.
            annotation_paths (str or list): Path(s) to ground truth dataset.
            k (int): Number of nearest neighbors for similarity evaluation.
        """
        self.verbose = verbose
        self.k = k  # the number of nearest neighbors
        self.tokenizer = tokenizer
        self.model = model
        self.real = []
        self.annotation_paths = annotation_paths if isinstance(annotation_paths, list) else [annotation_paths]
        self._build_real_dataset(annotation_paths)

        # Build nearest neighbor search model
        self.val_embeddings = np.array([entry["right_embedding"] for entry in self.real])
        self.nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        self.nn.fit(self.val_embeddings)

    def _log(self, message, level="info"):
        """
        Print log messages only if verbose mode is enabled.

        Args:
            message (str): The message to be logged.
            level (str): Log level, can be 'info', 'warning', or 'error'.
        """
        if self.verbose:
            prefix = {"info": "[INFO]", "warning": "[WARNING]", "error": "[ERROR]"}.get(level, "[INFO]")
            print(f"{prefix} {message}")

    def _load_model(self, finetuned_model_path):
        """
        Load the fine-tuned LLaMA model.

        Args:
            finetuned_model_path (str): Path to the fine-tuned model.

        Returns:
            tuple: (model, tokenizer)
        """
        self._log("Loading fine-tuned model...", level="info")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                finetuned_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

            tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
            tokenizer.pad_token = tokenizer.eos_token  # Ensure correct padding behavior

            self._log("Fine-tuned model loaded successfully!", level="info")
            return model, tokenizer

        except Exception as e:
            self._log(f"Error loading model: {e}", level="error")
            raise RuntimeError("Failed to load the fine-tuned model.") from e

    def _text_to_embedding(self, text, layer_idx=-3):
        """ Convert text into an embedding using LLM. """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_state = outputs.hidden_states[layer_idx]
        embedding = hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    def _build_real_dataset(self, annotation_paths):
        """
        Builds the ground truth dataset by extracting 'right_value' and 'error_value'
        from a JSONL file and converting them into embeddings.

        Args:
            annotation_paths (str): Path to the ground truth JSONL dataset.

        This function first checks for a cached version of the dataset to speed up loading.
        If the cache is not found, it processes the dataset, computes embeddings, and saves
        them for future use.
        """
        cache_file = "/home/liuxinyuan/table_tuning_for_error_generating_task/evaluation/real_errors_unique.json"

        # 1. Check if a cached version of the dataset exists
        if os.path.exists(cache_file):
            self._log(f"Found cached dataset at {cache_file}. Loading...", level="info")
            with open(cache_file, "r", encoding="utf-8") as f:
                self.real = json.load(f)

            # Convert list back to numpy arrays for efficient computations
            for item in self.real:
                item["right_embedding"] = np.array(item["right_embedding"])
                item["error_embedding"] = np.array(item["error_embedding"])

            self._log(f"Loaded {len(self.real)} real-world error samples.", level="info")
            return

        # 2. If no cache is found, process the dataset from scratch
        self.real = []

        # Count total lines in the dataset for progress tracking
        with open(annotation_paths, "r", encoding="utf-8-sig") as f:
            total_lines = sum(1 for _ in f)

        self._log(f"Processing dataset: {total_lines} records found.", level="info")

        # Process each record in the dataset and extract values
        with open(annotation_paths, "r", encoding="utf-8-sig") as f:
            for line in tqdm(f, total=total_lines, desc="Processing dataset", unit="samples", disable=not self.verbose):
                data = json.loads(line.strip())

                # Extract 'right_value' and 'error_value'
                data_output = data.get("output", "")
                right_value = data_output.get("right_value", "")
                error_value = data_output.get("error_value", "")

                # Ensure both values exist before processing
                if right_value and error_value:
                    right_embedding = self._text_to_embedding(right_value).tolist()
                    error_embedding = self._text_to_embedding(error_value).tolist()

                    self.real.append({
                        "right_value": right_value,
                        "error_value": error_value,
                        "right_embedding": right_embedding,
                        "error_embedding": error_embedding
                    })

        self._log(f"Successfully built the real-world error dataset with {len(self.real)} records.", level="info")

        # 3. Save processed data to a cache file for future use
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(self.real, f, ensure_ascii=False, indent=4)

        self._log(f"Cached dataset saved at {cache_file} for faster future loading.", level="info")

        # # self.real = self.real[:1000]
        #
        # # 5. Optionally save embeddings to a JSON file
        # embedding_file = "real_errors_embeddings.json"
        # save_embeddings = True  # Set to False if saving is not required
        #
        # if save_embeddings:
        #     embeddings = [
        #         {"right_embedding": item["right_embedding"], "error_embedding": item["error_embedding"]}
        #         for item in self.real
        #     ]
        #     with open(embedding_file, "w", encoding="utf-8") as f:
        #         json.dump(embeddings, f, ensure_ascii=False, indent=4)
        #     self._log(f"Embeddings saved to {embedding_file}.", level="info")

    def evaluate(self, generated_jsonl, generated_data=None):
        """
        Evaluate the quality of LLM-generated errors by comparing them with real-world errors.

        Args:
            generated_jsonl (str): Path to the JSONL file containing generated errors.
            generated_data (list, optional): Preloaded list of (right_value, error_value) tuples.

        Returns:
            tuple: (List of max cosine similarities, Mean similarity score)
        """
        """
        评估 LLM 生成错误 error_gene 的质量：
        1. 用 LLM Tokenizer 对 real_val 进行向量化
        2. 找到 self.real 里最近的 k 个真实值
        3. 计算 k 个 (real_val_embed - real_error_embed) 变换向量
        4. 计算 error_gene 的隐藏层表示 
        5. 计算 error_gene_embed 和 k 个变换向量的余弦相似度，返回均值
        """
        # Load generated errors if not provided
        if generated_data is None:
            generated_data = []
            with open(generated_jsonl, "r", encoding="utf-8-sig") as f:
                for line in f:
                    data = json.loads(line.strip())
                    error_value = data.get("error_value", "")
                    right_value = data.get("right_value", "")
                    if right_value and error_value:
                        generated_data.append((right_value, error_value))

            self._log(f"Loaded {len(generated_data)} generated error samples from {generated_jsonl}.", level="info")
        else:
            self._log(f"Using provided generated error dataset with {len(generated_data)} samples.", level="info")

        max_similarities = []

        for real_val, error_gene in tqdm(
            generated_data, desc="Evaluating errors", unit="samples", disable=not self.verbose
        ):
            if not isinstance(real_val, str) or not isinstance(error_gene, str):
                continue  # Skip invalid data types
            if real_val == error_gene:
                continue  # Skip cases where error == ground truth

            # Convert real value to embedding
            real_val_embed = self._text_to_embedding(real_val)

            # Find k-nearest real-world errors
            _, indices = self.nn.kneighbors([real_val_embed])
            indices = indices[0]

            # Compute transformation vectors (right_value - error_value)
            delta_vectors = np.array([
                self.real[idx]["right_embedding"] - self.real[idx]["error_embedding"]
                for idx in indices
            ])

            # Convert generated error to embedding
            error_gene_embed = self._text_to_embedding(error_gene)

            # Compute cosine similarity
            similarities = [1 - cosine(real_val_embed - error_gene_embed, delta) for delta in delta_vectors]
            max_similarities.append(max(similarities))


        mean_similarity = np.mean(max_similarities) if max_similarities else 0.0
        self._log(f"Evaluation completed. Mean similarity: {mean_similarity:.4f}", level="info")

        return max_similarities, mean_similarity

if __name__ == "__main__":
    # Define the base directory as the current script's location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(BASE_DIR, "llama3.1-8B")
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(BASE_DIR, "llama3.1-8B"),
        torch_dtype=torch.float32,  # Use float32, bf16 is not supported
        device_map="auto"  # Automatically select the device
    )

    # Path to the real-world error dataset (ground truth)
    real_annotation_file = os.path.join(BASE_DIR, "dataset/train/train_ErrorGeneration.jsonl")

    # Paths to the generated error datasets for different models
    generated_jsonl_my_model = os.path.join(BASE_DIR, "evaluation/test_dataset/exp_1/TableEG_output/beers/beers_20/error_log_20.jsonl")
    generated_jsonl_bart = os.path.join(BASE_DIR, "evaluation/test_dataset/exp_1/BART_output/beers/error_log.jsonl")
    generated_jsonl_GPT = os.path.join(BASE_DIR, "evaluation/test_dataset/exp_1/GPT_output/beers/beers_error_GPT.jsonl")

    print(f"Using {generated_jsonl_my_model} and {generated_jsonl_bart} for evaluation...")

    # Initialize ErrorQualityEvaluator and evaluate for different values of k
    for k in [1, 2, 3, 4, 5, 10, 15, 20]:
        print(f"🔍 Evaluating with k={k}...")

        evaluator = ErrorQualityEvaluator(tokenizer, model, real_annotation_file, k=k)

        # Evaluate errors generated by MyModel
        print("Evaluating MyModel-generated errors...")
        mymodel_similarities, mymodel_avg_similarity = evaluator.evaluate(generated_jsonl_my_model)
        print(f"TableEG Average Error Similarity: {mymodel_avg_similarity:.4f}")

        # Evaluate errors generated by BART
        print("\nEvaluating BART-generated errors...")
        bart_similarities, bart_avg_similarity = evaluator.evaluate(generated_jsonl=generated_jsonl_bart)
        print(f"BART Average Error Similarity: {bart_avg_similarity:.4f}")

        # Evaluate errors generated by GPT
        print("\nEvaluating GPT-generated errors...")
        gpt_similarities, gpt_avg_similarity = evaluator.evaluate(generated_jsonl=generated_jsonl_GPT)
        print(f"GPT Average Error Similarity: {gpt_avg_similarity:.4f}")