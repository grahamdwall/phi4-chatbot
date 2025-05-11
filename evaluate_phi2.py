from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import os
import time
import torch
import argparse
from datetime import datetime
import re
from math import isclose
from train_server import model_files_exist

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {DEVICE}")

timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = "../workspace/data/test/results"
os.makedirs(output_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
parser.add_argument("--test_set_path", type=str, required=True, help="Path to test set (JSONL)")
parser.add_argument("--tolerance", type=float, default=0.01, help="Relative tolerance for payment match")
args = parser.parse_args()

model_path = args.model_path
print(f"[DEBUG] Model path: {model_path}")
test_set_path = args.test_set_path
print(f"[DEBUG] Test set path: {test_set_path}")

print("[DEBUG] Starting evaluate_phi2.py... 🚀")
print(f"[DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"[DEBUG] torch.cuda.device_count(): {torch.cuda.device_count()}")
print(f"[DEBUG] Environment Variables: {os.environ}")

print("Current working directory:", os.getcwd())

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

def extract_payment(text):
    match = re.search(r"[\\$]?\\s*([\\d,]+\\.?\\d*)", text)
    return float(match.group(1).replace(",", "")) if match else None

def extract_first_number(text):
    match = re.search(r"([\d,]+\.?\d*)", str(text))
    if match:
        return float(match.group(1).replace(",", ""))
    return None

try:
    print(f"[DEBUG] Attempting to load model from {model_path}...")
    if os.path.isdir(model_path):
        if model_files_exist(model_path):
            #model_path = os.getenv("MODEL_PATH", "/workspace/data/models/microsoft/phi-2")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                #load_in_4bit=True,
                torch_dtype=torch.float32,
                #torch_dtype=dtype,
                local_files_only=True,
                trust_remote_code=True,
            )
            print("[DEBUG] Model loaded successfully ✅")
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                local_files_only=True, 
                trust_remote_code=True,
            )
            tokenizer.pad_token = tokenizer.eos_token  # Ensure padding doesn't crash it
            print("[DEBUG] Tokenizer loaded ✅")
            test_output = generate_text("Hello, world!")
            print(f"[DEBUG] Startup generation success: {test_output}")

            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

            # Read test data
            print("[DEBUG] Reading test data...")
            with open(test_set_path, "r") as f:
                test_data = [json.loads(line) for line in f]

            print(f"Evaluating on {len(test_data)} samples...")
            correct = 0
            total = 0
            start_time = time.time()

            for example in test_data:
                prompt = example.get("prompt") or example.get("instruction")
                print(f"[DEBUG] Prompt: {prompt}")
                expected = example.get("expected_payment") or example.get("output")
                print(f"[DEBUG] Expected: {expected}")
                if prompt is None or expected is None:
                    continue

                expected_val = extract_first_number(expected)

                response = pipe(prompt, max_new_tokens=50, do_sample=False)[0]["generated_text"]
                print(f"[DEBUG] Response: {response}")
                predicted_val = extract_first_number(response)

                if expected_val is None or predicted_val is None:
                    print(f"[WARN] Could not extract number from:\nExpected: {expected}\nGenerated: {response}")
                    continue

                print(f"[DEBUG] Predicted value: {predicted_val}")
                if predicted_val is not None and isclose(predicted_val, expected_val, rel_tol=args.tolerance):
                    correct += 1
                else:
                    print(f"❌ Prompt: {prompt}\nExpected: {expected_val} | Predicted: {predicted_val}\n")

                total += 1

            elapsed = time.time() - start_time
            if total == 0:
                print("⚠️ No valid examples found. Check the test set formatting.")
            else:
                print(f"✅ Accuracy: {correct}/{total} ({correct / total:.2%})")
            print(f"⏱ Completed in {elapsed:.2f} seconds")
        else:
            print(f"🛑 Model files not found in {model_path}. Please check the path.")
    else:
        print(f"🛑 Model path {model_path} is not a directory. Please check the path.")
except Exception as e:
    print(f"[ERROR] Exception during model loading: {str(e)}")
    raise
