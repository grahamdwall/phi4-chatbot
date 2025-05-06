import os
import torch
import json
import time
from fastapi import FastAPI, Request, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import uvicorn
from contextlib import asynccontextmanager
from pydantic import BaseModel

# python3 -m venv venv 
# source venv/bin/activate

# Set your paths
LOCAL_MODEL_PATH_PRETRAIN = "/workspace/data/models/phi2-base"
LOCAL_MODEL_PATH_FINETUNED = "/workspace/data/models/phi2-finetune"
HF_MODEL_PATH = "GrahamWall/phi2-finetune"
TRAINING_DATASETS = "./data/test/*.jsonl"

# Air-Gapped deployment: offline, cloud, serverless
print("[DEBUG] Starting train_phi2_lora.py... 🚀")
print(f"[DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"[DEBUG] torch.cuda.device_count(): {torch.cuda.device_count()}")
print(f"[DEBUG] Environment Variables: {os.environ}")

print("Current working directory:", os.getcwd())

# Check for local files
def model_files_exist(path):
    required_files = ["added_tokens.json", 
                      "config.json", 
                      "generation_config.json", 
                      "model-00001-of-00002.safetensors", 
                      "model-00002-of-00002.safetensors", 
                      "model.safetensors.index.json", 
                      "special_tokens_map.json", 
                      "tokenizer_config.json", 
                      "tokenizer.json", 
                      "vocab.json"]
    return all(os.path.isfile(os.path.join(path, f)) for f in required_files)

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32 if torch.cuda.is_available() else torch.float32
MAX_NUM_TOKENS = 100 if torch.cuda.is_available() else 1

print(f"[DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"[DEBUG] Dtype selected: {dtype}")

def generate_text(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=MAX_NUM_TOKENS)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / {total} ({100 * trainable / total:.2f}%)")

try:
    print("[DEBUG] Attempting to load model...")
    if os.path.isdir(LOCAL_MODEL_PATH_PRETRAIN) and model_files_exist(LOCAL_MODEL_PATH_PRETRAIN):
        #model_path = os.getenv("MODEL_PATH", "/workspace/data/models/microsoft/phi-2")
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH_PRETRAIN,
            device_map="auto",
            #load_in_4bit=True,
            torch_dtype=torch.float32,
            #torch_dtype=dtype,
            local_files_only=True,
            trust_remote_code=True,
        )
        print("[DEBUG] Model loaded successfully ✅")
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_PATH_PRETRAIN, 
            local_files_only=True, 
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token  # Ensure padding doesn't crash it
        print("[DEBUG] Tokenizer loaded ✅")
        test_output = generate_text("Hello, world!")
        print(f"[DEBUG] Startup generation success: {test_output}")
        # if we loaded from HF successfully, save to local path for next time
        #if model_path == HF_MODEL_PATH:
        #    # Create directory if it doesn't exist
        #    os.makedirs(LOCAL_MODEL_PATH_PRETRAIN, exist_ok=True)
        #    model.save_pretrained(LOCAL_MODEL_PATH_PRETRAIN)
        #    tokenizer.save_pretrained(LOCAL_MODEL_PATH_PRETRAIN)

        # Optional: cast to fp16 and prepare for PEFT
        # this requires 100MB of VRAM
        #model = prepare_model_for_kbit_training(model)

        # PEFT LoRA config
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "dense"],  # Make sure these match actual model module names
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        #peft_config = LoraConfig(
        #    r=8,
        #    lora_alpha=16,
        #    target_modules=["q_proj", "v_proj"],
        #    lora_dropout=0.1,
        #    bias="none",
        #    task_type=TaskType.CAUSAL_LM,
        #)

        model = get_peft_model(model, peft_config)

        # Optional: print # trainable params
        print_trainable_parameters(model)

        #for name, module in model.named_modules():
        #    print(name)

        print(f"Model hidden size: {model.config.hidden_size}")
        print(f"Peft config: r={peft_config.r}, alpha={peft_config.lora_alpha}")

        # Load your JSONL file into a HuggingFace dataset
        dataset = load_dataset("json", data_files=TRAINING_DATASETS, split="train")

        if len(dataset) > 0:

            print("Dataset columns:", dataset.column_names)

            required_cols = {"instruction", "output"}
            if not required_cols.issubset(dataset.column_names):
                raise ValueError(f"Dataset missing required columns: {required_cols}")

            # Tokenize
            def tokenize_fn(example):
                merged_text = f"{example['instruction']}\n{example['output']}"
                return tokenizer(merged_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")

            tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["instruction", "output"])

            # Training args
            training_args = TrainingArguments(
                output_dir=LOCAL_MODEL_PATH_FINETUNED,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                num_train_epochs=3,
                learning_rate=2e-4,
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,
                fp16=False,         # use float32
                bf16=False,        # explicitly avoid bfloat16
                report_to="none",
            )

            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized,
                data_collator=data_collator,
            )

            trainer.train()

except Exception as e:
    print(f"[ERROR] Exception during model loading: {str(e)}")
    raise

