import os
import sys
import time
import json
import torch
import subprocess
import threading
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse
from huggingface_hub import HfApi, HfFolder, snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from dotenv import load_dotenv
import glob
from contextlib import asynccontextmanager

# Global flag
model = None
tokenizer = None
model_ready = False
download_lock = threading.Lock()

TRAINING_SCRIPT = "train_phi2_lora.py"
DATASET_DIR = "../workspace/data/datasets"
TRAINING_LOG = "../workspace/training.log"
TESTING_SCRIPT = "evaluate_phi2.py"
TEST_SET_PATH_PRETRAIN = "./data/test/mortgage_finetune_1000.jsonl"
TEST_SET_PATH_FINETUNE = "./data/test/mortgage_finetune_1000.jsonl"
TEST_RESULTS_DIR = "../workspace/data/test/results"
TESTING_LOG = "../workspace/testing.log"
STATUS_FILE = "../workspace/training_status.json"
#MODEL_PATH = "/workspace/data/models/phi2-mortgage-lora"
#TOKENIZER_PATH = MODEL_PATH
LOCAL_MODEL_PATH_PRETRAIN = "../workspace/data/models/phi2-base"
LOCAL_MODEL_PATH_FINETUNED = "../workspace/data/models/phi2-finetune"
HF_MODEL_PATH = "GrahamWall/phi2-finetune"
local_files_only_computed = None
trust_remote_code_computed = None

model_path = LOCAL_MODEL_PATH_PRETRAIN
test_set_path = None

sys.stdout.reconfigure(line_buffering=True)

def background_download():
    global model, tokenizer, model_ready

    try:
        print(f"[DEBUG] Downloading model {model_path} to {LOCAL_MODEL_PATH_PRETRAIN}...")
        snapshot_download(
            repo_id=HF_MODEL_PATH,
            local_dir=LOCAL_MODEL_PATH_PRETRAIN,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print("[DEBUG] Model files downloaded ✅")

        print("[DEBUG] Loading model and tokenizer into memory...")
        with download_lock:
            model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH_PRETRAIN, device_map="auto", torch_dtype=torch.float32, local_files_only=True, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH_PRETRAIN, local_files_only=True, trust_remote_code=True)
            model_ready = True
        print("[DEBUG] Model and tokenizer loaded ✅")

    except Exception as e:
        print(f"[ERROR] Model download or load failed: {e}")
        model_ready = False

def get_missing_files(path):
    base_required = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    ]

    adapter_required = [
        "adapter_model.safetensors",
        "adapter_config.json",
    ]

    tokenizer_required = [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "added_tokens.json",
        "special_tokens_map.json",
    ]

    # Adapter-only case
    adapter_missing = [f for f in adapter_required if not os.path.isfile(os.path.join(path, f))]
    if not adapter_missing:
        return []  # All adapter files present, return empty list (no missing files)

    # Full model case: need to parse index file for shard names
    index_path = os.path.join(path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return base_required + tokenizer_required  # Missing index file: assume all base files missing

    try:
        import json
        with open(index_path, "r") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        shard_files = list(set(weight_map.values()))
    except Exception as e:
        print(f"[ERROR] Reading index.json failed: {e}")
        return base_required + tokenizer_required

    full_required = base_required + shard_files + tokenizer_required
    return [f for f in full_required if not os.path.isfile(os.path.join(path, f))]

def model_files_exist(path):
    missing = get_missing_files(path)
    if missing:
        print(f"🛑 Missing model files in {path}:")
        for f in missing:
            print(f" - {f}")
        return False
    print("✅ All required model files present.")
    return True

# Decide where to load from
if os.path.isdir(LOCAL_MODEL_PATH_PRETRAIN) and model_files_exist(LOCAL_MODEL_PATH_PRETRAIN):
    print(f"✅ Loading model from local path: {LOCAL_MODEL_PATH_PRETRAIN}")
    model_path = LOCAL_MODEL_PATH_PRETRAIN
    local_files_only_computed = True
    trust_remote_code_computed = True
else:
    print(f"⚠️ Local model not found, loading from Hugging Face: {HF_MODEL_PATH}")
    model_path = HF_MODEL_PATH
    local_files_only_computed = False
    trust_remote_code_computed = True

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32 if torch.cuda.is_available() else torch.float32
MAX_NUM_TOKENS = 100 if torch.cuda.is_available() else 1

#print(f"[DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}")
#print(f"[DEBUG] Dtype selected: {dtype}")

def generate_text(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=MAX_NUM_TOKENS)

# --- FASTAPI SETUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, model_ready
    print("[DEBUG] FastAPI lifespan started")
    print("[DEBUG] Starting up app")
    # Check if all required files exist
    if not model_files_exist(model_path):
        print("🛑 Model files not found. Starting background download...")
        threading.Thread(target=background_download, daemon=True).start()
    else:
        print(f"✅ Loading model from local path: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        print("[DEBUG] Model loaded successfully ✅")
        model_ready = True

    yield
    print("[DEBUG] Shutting down app")

app = FastAPI(lifespan=lifespan)
print("[DEBUG] FastAPI app created ✅")

# Add this CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your GitHub Pages domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("[DEBUG] CORS middleware added ✅")

pipe = None  # global declaration

@app.post("/chat")
async def chat(request: Request):
    global pipe, model_ready, model, tokenizer

    if not model_ready or model is None or tokenizer is None:
        return JSONResponse({"error": "Model not ready"}, status_code=503)
    
    if pipe is None:
        pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    data = await request.json()
    prompt = data.get("prompt", "")
    print(f"[DEBUG] Received prompt: {prompt}")
    if not prompt:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing 'prompt' in request body."}
        )

    try:
        result = pipe(prompt, max_new_tokens=256, do_sample=True)[0]["generated_text"]
        return {"response": result}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Model generation failed: {str(e)}"}
        )

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

def run_testing():
    global test_status, test_start_time, test_end_time, latest_results
    test_status = "testing_started"
    test_start_time = time.time()

    with open("testing.log", "w") as log_file:
        process = subprocess.Popen(
            ["accelerate", "launch", "evaluate_phi2.py", "--model_path", model_path, "--test_set_path", test_set_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            print(line, end="")             # stream to console
            log_file.write(line)            # write to log file
        process.wait()

    test_end_time = time.time()
    test_status = "testing_complete"

@app.post("/test-pretrain-performance")
def test_model_pretrain():
    global model_path, test_set_path
    global testing_proc

    if not model_ready:
        return {"error": "Model not ready yet. Try again later."}
    
    # Proceed with testing...
    if testing_proc and testing_proc.poll() is None:
        return {"error": "Testing already in progress"}

    model_path = LOCAL_MODEL_PATH_PRETRAIN
    test_set_path = TEST_SET_PATH_PRETRAIN
    thread = threading.Thread(target=run_testing)
    thread.start()
    return {"message": "Testing started in background"}

def get_latest_checkpoint(path):
    checkpoints = [d for d in os.listdir(path) if d.startswith("checkpoint-")]
    if not checkpoints:
        raise ValueError("No checkpoints found in path.")
    latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return os.path.join(path, latest)

@app.post("/test-finetune-performance")
def test_model_finetune():
    global testing_proc
    global model_path, test_set_path

    if not model_ready:
        return {"error": "Model not ready yet. Try again later."}
    
    # Proceed with testing...
    if testing_proc and testing_proc.poll() is None:
        return {"error": "Testing already in progress"}

    test_set_path = TEST_SET_PATH_FINETUNE
    model_path = get_latest_checkpoint(LOCAL_MODEL_PATH_FINETUNED)
    if model_path is None:
        return {"error": "No checkpoints found for finetuned model."}
    else:
        print(f"Using latest checkpoint: {model_path}")
        thread = threading.Thread(target=run_testing)
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
    # ssh the token to the mounted volume in RunPod instance, do not hardcode in source, or package with container
    load_dotenv("/workspace/.env")
    # Get the token from the environment variable
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

@app.get("/model_status")
def get_model_status():
    return {
        "ready": model_ready,
        "model_loaded": model is not None,
        "missing_files": get_missing_files(LOCAL_MODEL_PATH_PRETRAIN) if not model_ready else []
    }

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
