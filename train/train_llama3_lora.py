import os
import sys
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

# Disable NCCL P2P and IB to avoid communication issues
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current script's directory
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # Go up one level to project root

# Set dataset path (relative)
train_data_dir = os.path.join(PROJECT_ROOT, "dataset", "train")

# Set model path (relative)
model_path = os.path.join(PROJECT_ROOT, "llama3.1-8B")

# Set checkpoint directory (relative)
checkpoint_dir = os.path.join(PROJECT_ROOT, "output", "llama3.1_0221")

# Debugging: Print paths to verify correctness
print(f"Project Root: {PROJECT_ROOT}")
print(f"Train Data Directory: {train_data_dir}")
print(f"Model Path: {model_path}")
print(f"Checkpoint Directory: {checkpoint_dir}")

# Release GPU memory
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Configure PyTorch CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

# Load all JSONL files
all_data = []
for filename in os.listdir(train_data_dir):
    if filename.endswith(".jsonl"):
        file_path = os.path.join(train_data_dir, filename)
        print(f"Loading file: {filename}")
        with open(file_path, "r", encoding="utf-8") as f:
            all_data.extend([json.loads(line) for line in f])

# Initialize token length statistics
max_length = 0
lengths = []
total_tokens = 0

# Load LLaMA3 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# # Compute token length statistics
# print("Computing dataset statistics...")
# for example in all_data:
#     user_input = f"User:\n{example['instruction']}\nInput:\n{example['input']}\nAssistant:\n"
#     output = f"{example['output']}"

#     input_enc = tokenizer(user_input, add_special_tokens=True)
#     output_enc = tokenizer(output, add_special_tokens=False)

#     total_length = len(input_enc["input_ids"]) + len(output_enc["input_ids"])
#     lengths.append(total_length)
#     total_tokens += total_length
#     max_length = max(max_length, total_length)

# # Display token length statistics
# print(f"10% percentile: {np.quantile(lengths, 0.10)}")
# print(f"90% percentile: {np.quantile(lengths, 0.90)}")
# print(f"80% percentile: {np.quantile(lengths, 0.80)}")
# print(f"70% percentile: {np.quantile(lengths, 0.70)}")
# print(f"Max token length: {max_length}")
# print(f"Average token length: {sum(lengths) / len(lengths):.2f}")
# print(f"Total tokens: {total_tokens}")

# sys.exit(0)

# Convert JSON dictionary inputs to string format
for item in all_data:
    if isinstance(item.get("input"), dict):
        item["input"] = json.dumps(item["input"])
    if isinstance(item.get("output"), dict):
        item["output"] = json.dumps(item["output"])

# Convert dataset to Hugging Face Dataset format
dataset = Dataset.from_list(all_data)

# Display sample data
print("Sample data:", dataset[0])
print(f"Dataset size: {len(dataset)}")

# Define preprocessing function
def process_func(example):
    MAX_LENGTH = 3072  # Set token limit
    user_input = f"User:\n{example['instruction']}\nInput:\n{example['input']}\nAssistant:\n"
    output = f"{example['output']}"

    # Tokenization
    input_enc = tokenizer(user_input, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH)
    output_enc = tokenizer(output, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)

    # Combine input and output tokens
    input_ids = input_enc["input_ids"] + output_enc["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = input_enc["attention_mask"] + output_enc["attention_mask"] + [1]
    labels = [-100] * len(input_enc["input_ids"]) + output_enc["input_ids"] + [tokenizer.pad_token_id]

    return {
        "input_ids": input_ids[:MAX_LENGTH],
        "attention_mask": attention_mask[:MAX_LENGTH],
        "labels": labels[:MAX_LENGTH]
    }

# Apply preprocessing to dataset
dataset = dataset.map(process_func, remove_columns=dataset.column_names)

# Display sample preprocessed data
print("Sample preprocessed data:", tokenizer.decode(dataset[0]["input_ids"]))

# Load LLaMA3-8B model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
)

# Define LoRA configuration
lora_config = LoraConfig(
    task_type="CAUSAL_LM",  # Auto-regressive language modeling (GPT, LLaMA)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Transformer layers affected by LoRA
    inference_mode=False,
    r=32,  # LoRA rank
    lora_alpha=16,  # Scaling factor for LoRA layers
    lora_dropout=0.1  # Prevent overfitting
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Automatically detect the latest checkpoint
checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
if checkpoints:
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))  # Select the highest step
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"Resuming training from: {checkpoint_path}")
else:
    checkpoint_path = None
    print("No checkpoint found, starting from scratch.")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./output/llama3.1_0221",
    per_device_train_batch_size=1,  # Per GPU batch size
    gradient_accumulation_steps=16,  # Gradient accumulation steps
    save_steps=100,
    learning_rate=1e-4,
    weight_decay=0.02,  # L2 regularization strength in AdamW optimizer
    num_train_epochs=3,
    logging_steps=10,
    log_level="info",
    save_on_each_node=False,
    gradient_checkpointing=False,
    fp16=False,
    bf16=True,
    save_total_limit=3,
    report_to="none",  # Disable WandB or other logging tools
    ddp_find_unused_parameters=False,  # Improve multi-GPU training efficiency
    resume_from_checkpoint=bool(checkpoint_path)
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# Start training
print("Training started...")
if checkpoint_path:
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    trainer.train()

# Save fine-tuned LoRA model
peft_model_id = "./llama3.1_lora_config"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

print("LoRA fine-tuning completed. Model has been saved.")