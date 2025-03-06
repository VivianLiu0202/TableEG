import torch
import json
import random
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
import os
import sys

# sys.path.append("/home/liuxinyuan/table_tuning_for_error_generating_task/evaluation")
# ROOT = "/home/liuxinyuan/table_tuning_for_error_generating_task/evaluation"

# Set random seed for reproducibility
random.seed(42)

class ErrorGenerator:
    """A class for generating errors using LLaMA3 with LoRA fine-tuning."""


    def __init__(self, base_model_path, peft_model_path, dataset_dir, output_dir, verbose=True):
        """
        Initialize the error generator.

        Args:
            base_model_path (str): Path to the base LLaMA3 model.
            peft_model_path (str): Path to the LoRA fine-tuned weights.
            dataset_dir (str): Path to the dataset directory.
            output_dir (str): Path to the output directory.
            verbose (bool): Whether to print detailed logs.
        """
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.verbose = verbose
        self.model, self.tokenizer = self._load_model(base_model_path, peft_model_path)

        # Define operator mappings for constraints
        self.operator_map = {
            "EQ": "is equal to",
            "IQ": "is different from",
            "LT": "is less than",
            "GT": "is greater than",
            "LTE": "is less than or equal to",
            "GTE": "is greater than or equal to"
        }
        self.pattern = re.compile(r"(\w+)\(t\d+\.(\w+),t\d+\.(\w+)\)")

        # Define standard task descriptions
        self.task_descriptions = [
            "Modify a single attribute in the row identified by 'row_id' to introduce a {error_type} error. Ensure that the 'row_id' column is preserved as it uniquely identifies each record. Do NOT modify or generate new 'row_id' values.",
            "Introduce an error in a specific column of the row identified by 'row_id', creating a {error_type} error. The 'row_id' column must remain unchanged as it serves as the primary key, uniquely identifying each row, even if the table order is shuffled.",
            "Alter one column of a given row (selected using 'row_id') to create a {error_type} error. The row to be modified must be referenced using its 'row_id', which uniquely identifies each entry, regardless of the table order.",
            "Corrupt a single data entry in the row specified by 'row_id' to simulate a {error_type} error. Ensure 'row_id' is directly taken from the input table and remains unaltered.",
            "Modify a specific attribute in the row identified by 'row_id' to introduce a {error_type} error. Do NOT change or create new 'row_id' values.",
            "Change one column in the row determined by 'row_id' to generate a {error_type} error. The 'row_id' must be taken from the input table and remain unchanged.",
            "Introduce an error into a specific cell of the row identified by 'row_id', resulting in a {error_type} error. The 'row_id' serves as a fixed identifier and should not be modified."
        ]

        # Define error type descriptions
        self.error_type_descriptions = {
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

        # Define standard suffix descriptions
        self.suffix_descriptions = [
            "Provide the final result as a JSON object with 'row_id', 'column', 'error_type', 'error_value', 'right_value', 'missing_value', 'constraint', and 'tuple_pairs'. Select 'row_id' from the input table, modify only a single column, store the correct value in 'right_value', and the erroneous value in 'error_value'. Only include erroneous cell with high confidence.",
            "Output a JSON object containing 'row_id', 'column', 'error_type', 'error_value', 'right_value', 'missing_value', 'constraint', and 'tuple_pairs'. The 'row_id' must be from the input table, and the modified column’s correct value should be recorded in 'right_value', with the introduced error in 'error_value'. Report only error you are highly confident about.",
            "Generate a JSON result with 'row_id', 'column', 'error_type', 'error_value', 'right_value', 'missing_value', 'constraint', and 'tuple_pairs'. Ensure that 'row_id' is taken from the input table, the original correct value is placed in 'right_value', and the incorrect generated value is stored in 'error_value'. Include only high-confidence erroneous cell.",
            "Return a JSON object structured with 'row_id', 'column', 'error_type', 'error_value', 'right_value', 'missing_value', 'constraint', and 'tuple_pairs'. The 'row_id' must not be altered, and the original value should be saved in 'right_value' while the generated incorrect value should be placed in 'error_value'. Report only error with high certainty.",
            "Provide a JSON output including 'row_id', 'column', 'error_type', 'error_value', 'right_value', 'missing_value', 'constraint', and 'tuple_pairs'. The modified column's original data must be stored in 'right_value', and the generated incorrect entry should be recorded in 'error_value'. Only return error with high confidence.",
            "Output a structured JSON object with 'row_id', 'column', 'error_type', 'error_value', 'right_value', 'missing_value', 'constraint', and 'tuple_pairs'. The 'row_id' must come from the input table, and only a single column should be modified. Keep the correct value in 'right_value' and the incorrect generated value in 'error_value'. Only include confident error.",
            "Generate a JSON object containing 'row_id', 'column', 'error_type', 'error_value', 'right_value', 'missing_value', 'constraint', and 'tuple_pairs'. Ensure that 'row_id' is extracted from the input table, the modified column’s original value is recorded in 'right_value', and the erroneous value is stored in 'error_value'. Share only error with high certainty."
        ]

    def _log(self, message, level="info"):
        """Print messages only if verbose mode is enabled."""
        if self.verbose:
            prefix = {"info": "[INFO]", "warning": "[WARNING]", "error": "[ERROR]"}.get(level, "[INFO]")
            print(f"{prefix} {message}")

    def constraint_to_natural_language(self, dc_string):
        """Convert denial constraints into natural language descriptions."""
        predicates = []
        matches = self.pattern.findall(dc_string)
        for match in matches:
            operator, attr1, attr2 = match
            if operator in self.operator_map:
                predicates.append(f"The {attr1} of entity 1 {self.operator_map[operator]} the {attr2} of entity 2")
        return " and ".join(predicates) + "."

    def _load_model(self, base_model_path, peft_model_path):
        """
        Load the LLaMA3 model with LoRA fine-tuned weights.

        Args:
            base_model_path (str): Path to the base LLaMA3 model.
            peft_model_path (str): Path to the LoRA fine-tuned model.

        Returns:
            tuple: (LLaMA3 model, tokenizer)
        """
        try:
            self._log("Loading base model...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for efficient inference
                device_map="auto"  # Automatically select available devices
            )

            self._log("Loading LoRA fine-tuned weights...")
            model = PeftModel.from_pretrained(model, peft_model_path)  # Load LoRA weights
            model = model.merge_and_unload()  # Merge LoRA weights and unload unnecessary components

            self._log("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(peft_model_path)

            self._log("Model and tokenizer loaded successfully!")
            return model, tokenizer

        except Exception as e:
            self._log(f"Error loading model: {e}", level="error")
            raise RuntimeError("Failed to load the LLaMA3 model or tokenizer.") from e


    def generate_error_with_model(self, instruction, input_table):
        """
        Generate an error using the LLaMA3 model.

        Args:
            instruction (str): Instruction for the error generation task.
            input_table (str): Table data in a markdown format.

        Returns:
            dict or None: Parsed JSON output if successful, otherwise None.
        """
        self._log("Generating error using LLaMA3 model...")
        # Construct the prompt
        prompt = f"User:\n{instruction}\nInput:\n{input_table}\nAssistant:\n"
        try:
            # Encode the prompt using the tokenizer
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(self.model.device)  # Move to device
            attention_mask = inputs["attention_mask"].to(self.model.device)

            # Ensure the model is in evaluation mode
            self.model.eval()
            with torch.no_grad():  # Disable gradient calculation for inference
                output_ids = self.model.generate(
                    input_ids,
                    max_length=3072,
                    pad_token_id=self.tokenizer.pad_token_id,
                    attention_mask=attention_mask
                )

            # Decode the generated output
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Extract JSON content using regex
            match = re.search(r'Assistant:\s*(\{.*\})', output_text, re.DOTALL)
            model_output = match.group(1) if match else output_text

            # Parse JSON output
            parsed_output = json.loads(model_output)
            self._log("Error generated successfully.")
            return parsed_output

        except json.JSONDecodeError:
            self._log("JSON parsing failed. Returning raw text output.", level="warning")
            return None
        except Exception as e:
            self._log(f"Model generation error: {e}", level="error")
            return None

    def select_input_for_error(self, df, error_type, vio_pairs, used_rows=None):
        """
        Selects an appropriate table fragment for error generation.

        Args:
            df (pd.DataFrame): The original clean dataframe.
            error_type (str): Type of error to generate.
            vio_pairs (dict): Dictionary of potential rule violations.
            used_rows (set): Set of already used row IDs to avoid duplication.

        Returns:
            tuple: (Selected table as markdown, natural language description) or (None, None) if selection fails.
        """
        if error_type == "rule_violation":
            # Find potential rule violations
            rule, violation_candidates = self.find_potential_rule_violations(vio_pairs, used_rows)

            if rule is None or not violation_candidates:
                self._log("No valid rule violations found.", level="warning")
                return None, None

            # Convert rule to natural language
            nl_description = self.constraint_to_natural_language(rule)

            # Ensure violation_candidates is a list of strings
            violation_candidates = [str(row) for row in violation_candidates]

            if not violation_candidates:
                print("未找到违反约束的行！")
                return df.sample(n=5).to_markdown(index=False)  # 兜底策略，随机选取行

            # Retrieve all available row IDs that are not part of violation_candidates
            available_rows = [row for row in df["row_id"].unique() if row not in violation_candidates]

            # Select additional rows while ensuring the minimum required number
            num_additional_rows = min(3, len(available_rows))
            selected_rows = random.sample(available_rows, num_additional_rows) if num_additional_rows > 0 else []

            # Add the violation candidate rows
            selected_rows.extend(violation_candidates)
            random.shuffle(selected_rows)  # Shuffle to randomize order

            # Get the corresponding dataframe
            selected_df = df[df["row_id"].isin(selected_rows)]
            self._log(f"Selected {len(selected_rows)} rows for rule_violation.", level="info")

            return selected_df.to_markdown(index=False), nl_description
        else:
            # For other error types, randomly select 5 rows
            self._log(f"Selecting 5 random rows for {error_type} error generation.", level="info")
            return df.sample(n=5).to_markdown(index=False)

    def find_potential_rule_violations(self, vio_pairs, used_rows):
        """
        Identify potential rule violations while ensuring each pair is selected only once.

        Args:
            vio_pairs (dict): Dictionary of rule violations where keys are rules and values are lists of row pairs.
            used_rows (set): Set of already used row IDs to avoid duplicate selections.

        Returns:
            tuple: (Selected rule, violating row pair) or (None, None) if no valid pair is found.
        """
        # Retrieve all available rules
        rules = list(vio_pairs.keys())
        if not rules:
            self._log("No rules available for violation selection.", level="warning")
            return None, None

        # Randomly select a rule
        selected_rule = random.choice(rules)
        vio_context = vio_pairs[selected_rule]

        if not vio_context:
            self._log(f"Rule {selected_rule} has no remaining violating row pairs.", level="warning")
            return None, None

        # Try to find a valid row pair that has not been used
        while vio_context:
            selected_pair = random.choice(vio_context)

            if selected_pair[0] not in used_rows and selected_pair[1] not in used_rows:
                self._log(f"Selected rule violation pair: {selected_pair}.", level="info")
                break  # Valid pair found
            else:
                self._log(f"Pair {selected_pair} contains already used rows. Retrying...", level="warning")
                vio_pairs[selected_rule].remove(selected_pair)

        else:
            self._log(f"No valid pairs left for rule {selected_rule}.", level="error")
            return None, None

        # Remove the selected pair from available choices
        vio_pairs[selected_rule].remove(selected_pair)

        # If all pairs for the rule are exhausted, remove the rule from the dictionary
        if not vio_pairs[selected_rule]:
            del vio_pairs[selected_rule]
            self._log(f"All pairs for rule {selected_rule} have been used. Removing rule.", level="info")

        return selected_rule, selected_pair


    def generate_instruction(self, error_type, rule=None):
        """
        Generate a complete instruction, including task description, error type, and suffix.

        Args:
            error_type (str): Type of error (e.g., 'rule_violation', 'pattern_violation', etc.).
            rule (str, optional): Constraint rule, if applicable.

        Returns:
            str: A formatted instruction string.
        """
        task_desc = random.choice(self.task_descriptions).format(error_type=error_type)
        error_desc = random.choice(self.error_type_descriptions.get(error_type, ["Unknown error type."]))
        suffix_desc = random.choice(self.suffix_descriptions)


        if rule:
            constraint_desc = f"Denial Constraint is: {rule}."
            instruction = f"{task_desc} {error_desc} {constraint_desc} {suffix_desc}"
        else:
            instruction = f"{task_desc} {error_desc} {suffix_desc}"

        self._log(f"Generated instruction for {error_type} error.", level="info")
        return instruction

    def extract_tuple_pairs(self, error_json):
        """
        Extracts tuple_pairs and related columns from the error JSON.

        Args:
            error_json (dict): JSON object containing error information.

        Returns:
            set: A set of (row_id, column) pairs where errors exist.
        """
        self._log("Extracting tuple pairs from error JSON...", level="info")

        # Retrieve tuple_pairs and constraints
        tuple_pairs = error_json.get("tuple_pairs")
        constraint = error_json.get("constraint", "")

        if not tuple_pairs:
            self._log("No tuple_pairs found in the error JSON.", level="warning")
            return set()

        try:
            # Extract row IDs from tuple_pairs, e.g., "(698, 704)" -> {698, 704}
            row_ids = set(map(int, re.findall(r"\d+", tuple_pairs)))
        except ValueError:
            self._log("Error extracting row IDs from tuple_pairs.", level="error")
            return set()

        # Extract column names mentioned in the constraint
        column_matches = re.findall(r"The (\w+) of Row \d+", constraint)
        columns = set(column_matches)

        if not columns:
            self._log("No valid columns found in the constraint description.", level="warning")

        # Generate (row_id, column) pairs
        row_column_pairs = {(row, col) for row in row_ids for col in columns}

        self._log(f"Extracted {len(row_column_pairs)} row-column pairs.", level="info")
        return row_column_pairs


    def generate_errors_across_dataset(self, df, latent_vio_path, error_ratio=0.1, error_type_ratios="1:1:1:1",
                                       error_log_path="error_log.jsonl"):
        """
        Generate errors across the entire dataset while controlling the proportion of four error types.

        Args:
            df (pd.DataFrame): The original clean dataset.
            latent_vio_path (str): Path to JSON file containing precomputed rule violation row pairs.
            error_ratio (float): The proportion of errors to introduce into the dataset.
            error_type_ratios (str): Ratio of different error types (e.g., "1:1:3:2").
            error_log_path (str): Path to save the generated error log.

        Returns:
            tuple: (Path to the error log file, Set of modified cells).
        """
        total_cells = df.shape[0] * df.shape[1]
        total_errors = int(error_ratio * total_cells)

        self._log(f"Total dataset cells: {total_cells}")
        self._log(f"Planned total errors: {total_errors}")

        # Define error types and calculate distribution based on the given ratio
        error_types = ["outliers", "missing_value", "pattern_violation", "rule_violation"]
        ratios = list(map(int, error_type_ratios.split(":")))
        total_ratio = sum(ratios)

        # Compute the number of errors per type
        error_counts = {error_types[i]: int(total_errors * (ratios[i] / total_ratio)) for i in range(4)}
        self._log(f"Planned error distribution: {error_counts}")

        # Initialize tracking variables
        generated_errors = {error: 0 for error in error_types}
        used_cells = set()
        used_rows = set()
        error_list = []

        # Load precomputed violation row pairs
        with open(latent_vio_path, "r", encoding="utf-8") as f:
            vio_pairs = json.load(f)

        # Define the output error log path
        error_log_path = os.path.join(self.output_dir, error_log_path)
        self._log(f"Saving error log to: {error_log_path}")

        # Open file for writing errors
        with tqdm(total=total_errors, desc="Generating errors") as pbar, open(error_log_path, "w", encoding="utf-8") as log_file:
            while sum(generated_errors.values()) < total_errors:
                # Filter error types that have not yet reached their limit
                available_error_types = [e for e in error_types if generated_errors[e] < error_counts[e]]

                if not available_error_types:
                    self._log("All error types have reached their limit. Stopping generation.", level="info")
                    break

                # Randomly select an error type from available options
                error_type = random.choice(available_error_types)
                self._log(f"Selected error type: {error_type}")

                # Select input table and generate instruction
                if error_type == "rule_violation":
                    input_table, constraint = self.select_input_for_error(df, error_type, vio_pairs, used_rows)
                    if input_table is None and constraint is None:
                        generated_errors[error_type] = error_counts[error_type]  # Mark this type as complete
                        self._log("All rule violation constraints exhausted.", level="warning")
                        continue
                    instruction = self.generate_instruction(error_type, rule=constraint)
                else:
                    input_table = self.select_input_for_error(df, error_type, vio_pairs, used_rows)
                    instruction = self.generate_instruction(error_type)

                # Generate error using the model
                error_json = self.generate_error_with_model(instruction, input_table)
                if not error_json:
                    self._log("Generated error JSON is empty. Skipping...", level="warning")
                    continue

                # Extract necessary fields from model output
                error_value = str(error_json.get("error_value", "")).strip()
                right_value = str(error_json.get("right_value", "")).strip()
                column = error_json.get("column")
                row = error_json.get("row_id")

                # Validate extracted values
                if column not in df.columns:
                    self._log(f"Error generated for undefined column '{column}'. Skipping...", level="warning")
                    continue

                if row not in df["row_id"].values:
                    self._log(f"Error generated for undefined row '{row}'. Skipping...", level="warning")
                    continue

                # Retrieve real value from dataframe
                real_value = df.loc[df["row_id"] == row, column].values[0]
                self._log(f"Validating: error_value={error_value}, right_value={right_value}, real_value={real_value}")

                # Convert numerical values where applicable
                try:
                    real_value = float(real_value)
                    right_value = float(right_value)
                    error_value = float(error_value)
                except ValueError:
                    pass  # Keep values as strings if conversion fails

                # Ensure that right_value matches real_value; otherwise, correct it
                if right_value != real_value:
                    self._log(f"Mismatch detected: Updating right_value from {right_value} to {real_value}", level="info")
                    error_json["right_value"] = real_value

                # Ensure error_value is actually different from right_value
                if error_value == right_value:
                    self._log("Generated error value is identical to the correct value. Skipping...", level="warning")
                    continue

                # Process rule_violation errors separately
                if error_json.get("error_type") == "rule_violation":
                    rule_violation_cells = self.extract_tuple_pairs(error_json)
                    used_rows.add(row)
                    if not rule_violation_cells:
                        used_rows.remove(row)
                        self._log("No valid rule violation cells found. Skipping...", level="warning")
                        continue

                    # Ensure no duplicate errors on the same (row, column)
                    if rule_violation_cells & used_cells:
                        self._log("Duplicate (row, column) error detected. Skipping...", level="warning")
                        continue
                    used_cells.update(rule_violation_cells)
                else:
                    # Handle standard errors
                    if (row, column) in used_cells:
                        self._log("Duplicate (row, column) error detected. Skipping...", level="warning")
                        continue
                    used_cells.add((row, column))

                # Store the generated error
                error_list.append(error_json)
                log_file.write(json.dumps(error_json, ensure_ascii=False) + "\n")

                # Update counters
                generated_errors[error_type] += 1
                pbar.update(1)

                if pbar.n % 100 == 0:
                    self._log(f"Progress: Used rows count={len(used_rows)}", level="info")

        self._log(f"Error generation complete. Log saved to {error_log_path}", level="info")
        return error_log_path, used_cells


    def apply_errors_to_dataframe(self, df: pd.DataFrame, error_list: list, used_cells: set) -> pd.DataFrame:
        """
        Apply generated errors to the DataFrame while preventing duplicate modifications.

        Args:
            df (pd.DataFrame): The original clean DataFrame.
            error_list (list): List of error JSON strings generated by the model.
            used_cells (set): A set of (row_id, column) pairs that have already been modified.

        Returns:
            pd.DataFrame: The modified DataFrame containing injected errors.
        """
        error_count = 0

        self._log("Applying errors to the DataFrame...", level="info")

        for line in error_list:
            try:
                error_json = json.loads(line)
                row, column, error_value = error_json.get("row_id"), error_json.get("column"), error_json.get(
                    "error_value")

                # Validate row_id and column existence
                if row not in df["row_id"].values:
                    self._log(f"Row ID {row} not found in DataFrame. Skipping...", level="warning")
                    continue

                if column not in df.columns:
                    self._log(f"Column '{column}' not found in DataFrame. Skipping...", level="warning")
                    continue

                # Ensure the cell has not already been modified
                if (row, column) in used_cells:
                    self._log(f"Skipping duplicate modification for row_id={row}, column={column}", level="warning")
                    continue

                # Apply the error
                df.loc[df["row_id"] == row, column] = error_value
                used_cells.add((row, column))  # Mark cell as modified
                error_count += 1

                self._log(
                    f"Modified row_id={row}, column={column}, new_value={error_value}. Total errors applied: {error_count}",
                    level="info")

            except json.JSONDecodeError:
                self._log("Failed to decode JSON error entry. Skipping...", level="error")
                continue
            except Exception as e:
                self._log(f"Unexpected error while applying errors: {e}", level="error")
                continue

        self._log(f"Completed applying errors. Total errors injected: {error_count}", level="info")
        return df

if __name__ == "__main__":
    # Get the current working directory
    current_dir = os.getcwd()
    print(f"[INFO] Current working directory: {current_dir}")

    # Define relative paths
    base_model_path = os.path.join(current_dir, "models", "llama3.1-8B")
    peft_model_path = os.path.join(current_dir, "models", "llama3.1_lora_all_0221")
    dataset_dir = os.path.join(current_dir, "evaluation", "test_dataset", "exp_1", "raw_test_dataset")
    output_dir = os.path.join(current_dir, "evaluation", "test_dataset", "exp_1", "TableEG_output")
    clean_file = os.path.join(dataset_dir, "test_beers.csv")
    latent_vio_path = os.path.join(dataset_dir, "row_pairs_beers.json")
    error_log_path = os.path.join(output_dir, "beers", "beers_20", "error_log_20.jsonl")
    dirty_output_path = os.path.join(output_dir, "beers", "beers_20", "dirty_beers.csv")  # 新增: 脏数据集路径

    # Initialize the error generator
    error_generator = ErrorGenerator(base_model_path, peft_model_path, dataset_dir, output_dir, verbose=True)

    # Load clean dataset
    df_clean = pd.read_csv(clean_file, dtype=str)

    # Generate errors
    error_log_path, used_cells = error_generator.generate_errors_across_dataset(
        df_clean,
        latent_vio_path,
        error_ratio=0.2,
        error_type_ratios="1:2:4:3",
        error_log_path=error_log_path
    )
    print(f"[INFO] Error log generated and saved at: {error_log_path}")

    # ===============================
    # Read JSONL file and apply errors to the dataset
    # ===============================
    with open(error_log_path, "r", encoding="utf-8") as f:
        error_list = f.readlines()

    # Apply errors to `df_clean` to generate the `dirty` version of the dataset
    df_dirty = error_generator.apply_errors_to_dataframe(df_clean, error_list, used_cells)

    # Save the `dirty` dataset
    df_dirty.to_csv(dirty_output_path, index=False)
    print(f"[INFO] Dirty dataset successfully saved at: {dirty_output_path}")
