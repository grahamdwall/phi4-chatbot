from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import os
import time
import torch
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = "/workspace/data/test/results"
os.makedirs(output_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--test_set_path", type=str, required=True)
args = parser.parse_args()

model_path = args.model_path
test_set_path = args.test_set_path

model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if DEVICE == "cuda" else -1)

with open(test_set_path, "r") as f:
    lines = [json.loads(line) for line in f]

results = []
for ex in lines:
    prompt = ex["prompt"]
    expected = ex["expected"]
    output = pipe(prompt, max_new_tokens=64)[0]["generated_text"]
    match = expected.strip().lower() in output.strip().lower()
    results.append({"prompt": prompt, "expected": expected, "output": output, "match": match})

match_rate = sum(r["match"] for r in results) / len(results)

out_path = os.path.join(output_dir, f"eval_{timestamp}.json")
with open(out_path, "w") as f:
    json.dump({"match_rate": match_rate, "examples": results[:10]}, f, indent=2)

print(f"Saved evaluation results to {out_path}")
