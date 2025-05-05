import os
import time
import json
import subprocess
import threading
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse
from huggingface_hub import HfApi, HfFolder
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import glob

app = FastAPI()

TRAINING_SCRIPT = "train_phi2_lora.py"
DATASET_DIR = "/workspace/data/datasets"
TRAINING_LOG = "/workspace/training.log"
TESTING_SCRIPT = "evaluate_phi2_lora.py"
TEST_SET_PATH_PRETRAIN = "/workspace/data/test/mortgage_finetune_1000.jsonl"
TEST_SET_PATH_FINETUNE = "/workspace/data/test/mortgage_finetune_1000.jsonl"
TEST_RESULTS_DIR = "/workspace/data/test/results"
TESTING_LOG = "/workspace/testing.log"
STATUS_FILE = "/workspace/training_status.json"
MODEL_PATH = "/workspace/data/models/phi2-mortgage-lora"
TOKENIZER_PATH = MODEL_PATH
LOCAL_MODEL_PATH_PRETRAIN = "/workspace/data/models/phi2-base"
LOCAL_MODEL_PATH_FINETUNED = "/workspace/data/models/phi2-finetune"

# ssh the token to the mounted volume in RunPod instance, do not hardcode in source, or package with container
load_dotenv("/workspace/.env")

# Global state
training_proc: Optional[subprocess.Popen] = None
testing_proc: Optional[subprocess.Popen] = None

@app.post("/upload-training-data")
async def upload_dataset(file: UploadFile = File(...)):
    os.makedirs(DATASET_DIR, exist_ok=True)
    dest_path = os.path.join(DATASET_DIR, file.filename)
    with open(dest_path, "wb") as f:
        f.write(await file.read())
    return {"message": f"Uploaded {file.filename}"}

def run_training():
    global training_proc
    start_time = time.time()
    with open(STATUS_FILE, "w") as f:
        json.dump({"status": "training_started", "start_time": start_time}, f)

    with open(TRAINING_LOG, "w") as log_file:
        training_proc = subprocess.Popen(
            ["accelerate", "launch", TRAINING_SCRIPT],
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        training_proc.wait()

    end_time = time.time()
    duration = round(end_time - start_time, 2)
    with open(STATUS_FILE, "w") as f:
        json.dump({
            "status": "training_complete",
            "start_time": start_time,
            "end_time": end_time,
            "duration_sec": duration
        }, f)

# curl -X POST http://157.157.221.29:34378/train \
#  -H "Content-Type: application/json" \
@app.post("/train")
def start_training():
    global training_proc
    if training_proc and training_proc.poll() is None:
        return {"error": "Training already in progress"}

    thread = threading.Thread(target=run_training)
    thread.start()
    return {"message": "Training started in background"}

def run_testing(model_path: str):
    global testing_proc
    start_time = time.time()
    with open(STATUS_FILE, "w") as f:
        json.dump({"status": "testing_started", "start_time": start_time}, f)

    with open(TESTING_LOG, "w") as log_file:
        testing_proc = subprocess.Popen(
            ["accelerate", "launch", TESTING_SCRIPT, "--model_path", model_path],
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        testing_proc.wait()

    end_time = time.time()
    duration = round(end_time - start_time, 2)
    with open(STATUS_FILE, "w") as f:
        json.dump({
            "status": "testing_complete",
            "start_time": start_time,
            "end_time": end_time,
            "duration_sec": duration
        }, f)

@app.post("/test-pretrain-performance")
def test_model_pretrain():
    global testing_proc
    if testing_proc and testing_proc.poll() is None:
        return {"error": "Testing already in progress"}

    thread = threading.Thread(target=run_testing, args=(LOCAL_MODEL_PATH_PRETRAIN, TEST_SET_PATH_PRETRAIN))
    thread.start()
    return {"message": "Testing started in background"}

@app.post("/test-finetune-performance")
def test_model_finetune():
    global testing_proc
    if testing_proc and testing_proc.poll() is None:
        return {"error": "Testing already in progress"}

    thread = threading.Thread(target=run_testing, args=(LOCAL_MODEL_PATH_FINETUNED, TEST_SET_PATH_FINETUNE))
    thread.start()
    return {"message": "Testing started in background"}

# curl http://localhost:8080/get-latest-results
@app.get("/get-latest-results")
def get_latest_eval_results():
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    result_files = sorted(glob.glob(os.path.join(TEST_RESULTS_DIR, "eval_*.json")), reverse=True)

    if not result_files:
        return {"error": "No evaluation results found."}

    with open(result_files[0], "r") as f:
        result_data = json.load(f)

    return {
        "file": os.path.basename(result_files[0]),
        "result": result_data
    }

# Usage:
# Send a POST to /push-model with JSON body: { "hub_repo": "GrahamWall/phi2-finetune" }
@app.post("/push-model")
def push_model(hub_repo: str):
    """Push trained model and tokenizer to Hugging Face Hub"""
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return {"error": "Missing HUGGINGFACE_HUB_TOKEN env var"}

    # Optional login (sets ~/.huggingface/token)
    HfFolder.save_token(token)
    api = HfApi()

    try:
        print("Pushing model to:", hub_repo)
        AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH_FINETUNED).push_to_hub(hub_repo)
        AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH_FINETUNED).push_to_hub(hub_repo)
        return {"message": f"Model and tokenizer pushed to {hub_repo}"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/status")
def get_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    return {"status": "no_training_started"}

@app.get("/logs")
def get_logs():
    if os.path.exists(TRAINING_LOG):
        with open(TRAINING_LOG, "r") as f:
            return {"log": f.read()}
    return {"log": "No logs available."}

# e.g. curl http://157.157.221.29:34378/health
@app.get("/health")
def health_check():
    return {"status": "ok"}

# e.g. curl http://157.157.221.29:34378/
@app.get("/")
def read_root():
    return {"message": "LLM server is running"}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    print("[ERROR]", traceback.format_exc())
    return JSONResponse(status_code=500, content={"error": str(exc)})

# note: this code will not run in RunPod since we start uvicorn from the dockerfile
# but it is useful for local testing
# will start server using gunicorn as an external process in production
if __name__ == "__main__":
    import uvicorn
    print("[DEBUG] Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
