import os
import sys
import time
import json
import torch
import subprocess
import threading
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any, TypedDict, Union, Tuple
from fastapi.responses import JSONResponse
from huggingface_hub import HfApi, HfFolder, snapshot_download
from transformers import AutoModelForCausalLM, TextGenerationPipeline, AutoProcessor, GenerationConfig, AutoTokenizer
from dotenv import load_dotenv
import glob
import contextlib
from contextlib import asynccontextmanager
from uuid import uuid4
import re
import traceback
from mortgage_convo_fsm import MortgageConversation
from mortgage_rates import get_interest_rate_percent, calculate_cmhc_insurance, calculate_min_downpayment
import locationtagger
import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('treebank')
nltk.download('maxent_treebank_pos_tagger')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from urllib.parse import urlparse
import textwrap
import random, string
from threading import Lock

# Add the directory containing 'transformers_modules' to sys.path
#custom_code_path = os.path.join(
#    "../workspace/data/models/Phi-4-multimodal-instruct", "modules"
#)
#sys.path.insert(0, custom_code_path)

# Now import the model-specific module
#from transformers_modules.Phi_4_multimodal_instruct.modeling_phi4mm import InputMode

# python -m spacy download en_core_web_sm

# git add .
# git commit -m "Describe your changes here"
# git push origin main

# Global flag
model = None
processor = None
generation_config = None
tokenizer = None
model_ready = False
pipe = None
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
LOCAL_MODEL_PATH_PRETRAIN = "../workspace/data/models/Phi-4-multimodal-instruct"
#LOCAL_MODEL_PATH_PRETRAIN = "../workspace/data/models/Phi-4-mini-reasoning"
LOCAL_MODEL_PATH_FINETUNED = "../workspace/data/models/phi2-finetune"
#HF_MODEL_PATH = "microsoft/Phi-4-mini-reasoning"
HF_MODEL_PATH = "microsoft/Phi-4-multimodal-instruct"
#HF_MODEL_PATH = "GrahamWall/phi2-finetune"
local_files_only_computed = None
trust_remote_code_computed = None

model_path = LOCAL_MODEL_PATH_PRETRAIN
test_set_path = None

sys.stdout.reconfigure(line_buffering=True)

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def background_download():
    global model, tokenizer, processor, generation_config, model_ready

    try:
        print(f"[DEBUG] Downloading model {model_path} to {LOCAL_MODEL_PATH_PRETRAIN}...")
        snapshot_download(
            repo_id=HF_MODEL_PATH,
            local_dir=LOCAL_MODEL_PATH_PRETRAIN,
            force_download=False,
        )
        print("[DEBUG] Model files downloaded ✅")

        print("[DEBUG] Loading model and tokenizer into memory...")
        with download_lock:
            config_path = os.path.join(LOCAL_MODEL_PATH_PRETRAIN, "config.json")

            # --- Patch the config to disable flash_attn before loading ---
            with open(config_path, "r") as f:
                config = json.load(f)

            config["_attn_implementation"] = "eager"

            # Save modified config
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            with suppress_stdout():
                processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH_PRETRAIN, trust_remote_code=True)
            #print(processor.tokenizer)
            with suppress_stdout():
                model = AutoModelForCausalLM.from_pretrained(
                    LOCAL_MODEL_PATH_PRETRAIN,
                    device_map="cuda", 
                    torch_dtype="auto", 
                    trust_remote_code=True,
                    attn_implementation="eager",  # << disables flash-attn
                    #_attn_implementation='flash_attention_2',
                ).cuda()
            #print("model.config._attn_implementation:", model.config._attn_implementation)
            with suppress_stdout():
                tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH_PRETRAIN, trust_remote_code=True)
            # Load generation config
            generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')
            model_ready = True
        print("[DEBUG] Model and tokenizer loaded ✅")

    except Exception as e:
        print(f"[ERROR] Model download or load failed: {e}")
        model_ready = False

def get_missing_files(path):
    # Core model files
    base_required = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    ]

    # Tokenizer files
    tokenizer_required = [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "added_tokens.json",
        "special_tokens_map.json",
    ]

    # Multimodal-specific files
    multimodal_required = [
        "preprocessor_config.json",
        "image_processor_config.json",           # often used for vision transformers
        "open_clip_pytorch_model.bin",           # if used by Phi-4-mm
    ]

    # Check if adapter-only setup
    adapter_required = [
        "adapter_model.safetensors",
        "adapter_config.json",
    ]
    adapter_missing = [f for f in adapter_required if not os.path.isfile(os.path.join(path, f))]
    if not adapter_missing:
        return []  # Adapter files are present, return empty list

    # Full model case: check index for shard files
    index_path = os.path.join(path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        # Can't find index: assume all base+tokenizer+multimodal files are needed
        return base_required + tokenizer_required + multimodal_required

    try:
        with open(index_path, "r") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        shard_files = list(set(weight_map.values()))
    except Exception as e:
        print(f"[ERROR] Reading index.json failed: {e}")
        return base_required + tokenizer_required + multimodal_required

    # Compose full expected file list
    full_required = base_required + tokenizer_required + multimodal_required + shard_files
    return [f for f in full_required if not os.path.isfile(os.path.join(path, f))]

def is_multimodal_model(config_path):
    if not os.path.isfile(config_path):
        return False
    with open(config_path) as f:
        config = json.load(f)
    return 'vision_config' in config or 'image_processor_type' in config

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
device = "cuda"
dtype = torch.float32
#MAX_NUM_TOKENS = 2048   # limitation of Phi-2

#print(f"[DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}")
#print(f"[DEBUG] Dtype selected: {dtype}")

#def generate_text(prompt: str) -> str:
#    inputs = tokenizer(prompt, return_tensors="pt")
#    inputs = {k: v.to(model.device) for k, v in inputs.items()}
#    outputs = model.generate(**inputs, max_new_tokens=MAX_NUM_TOKENS)

# --- FASTAPI SETUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, processor, generation_config, model_ready
    print("[DEBUG] FastAPI lifespan started")
    print("[DEBUG] Starting up app")
    # Check if all required files exist
    download_started = False
    if not model_files_exist(model_path):
        print("🛑 Model files not found. Starting background download...")
        threading.Thread(target=background_download, daemon=True).start()
        download_started = True
    else:
        print(f"✅ Loading model from local path: {model_path}")
        config_path = os.path.join(LOCAL_MODEL_PATH_PRETRAIN, "config.json")

        # --- Patch the config to disable flash_attn before loading ---
        with open(config_path, "r") as f:
            config = json.load(f)

        config["_attn_implementation"] = "eager"

        # Save modified config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        with suppress_stdout():
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        #print(processor.tokenizer)
        with suppress_stdout():
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="cuda", 
                torch_dtype="auto", 
                trust_remote_code=True,
                attn_implementation="eager",  # << disables flash-attn
                #_attn_implementation='flash_attention_2',
            ).cuda()
        #print("model.config._attn_implementation:", model.config._attn_implementation)
        with suppress_stdout():
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Load generation config
        generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

        print("[DEBUG] Model loaded successfully ✅")
        model_ready = True

    # Wait for model to be ready if it was downloading
    if download_started:
        print("[DEBUG] Waiting for background model download to finish...")
        for _ in range(60*60):  # wait up to 1 hour (could be throttled)
            if model_ready:
                print("[DEBUG] Model download complete ✅")
                break
            time.sleep(1)
        else:
            print("[ERROR] Model did not become ready in time.")
            raise RuntimeError("Model failed to load during app startup")
        
    yield
    print("[DEBUG] Shutting down app")

app = FastAPI(lifespan=lifespan)
print("[DEBUG] FastAPI app created ✅")

# Conversation data structure with timestamps and FSM
#class SessionData(TypedDict, total=False):
#    history: list
#    last_updated: float
#    fsm: MortgageConversation

#conversation_state: Dict[str, SessionData] = {}

# In-memory session storage
sessions = {}
session_lock = Lock()

# Lock for thread-safe operations
#conversation_lock = threading.Lock()

# Session timeout in seconds (60 mins)
SESSION_TIMEOUT_SECONDS = 60 * 60

def cleanup_inactive_sessions():
    """Background thread to periodically clean up inactive sessions."""
    now = time.time()
    expired_ids = []
    with session_lock:
        for session_id, session in sessions.items():
            if now - session.get("last_updated", 0) > SESSION_TIMEOUT_SECONDS:
                expired_ids.append(session_id)
        for session_id in expired_ids:
            del sessions[session_id]
            print(f"[INFO] Session {session_id} expired and was removed.")    

# Add this CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("[DEBUG] CORS middleware added ✅")

def extract_price(text: str) -> Optional[float]:
    """
    Extracts the property price from the text.
    """
    match = re.search(r"(?:\$|\bprice(?: amount)? of )\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+)", text, re.IGNORECASE)
    if match:
        start, end = match.span()
        result = float(match.group(1).replace(",", ""))
        return {"value": result, "span": (start, end)}
    return None


def is_likely_url(text: str) -> bool:
    parsed = urlparse(text)
    return all([parsed.scheme, parsed.netloc])

# sample_text = "I recently moved from Toronto to Vancouver, and I plan to visit Montreal and Ottawa next month."
# locations = extract_locations(sample_text)
# print(locations)
def extract_locations(text: str) -> Dict[str, List[Dict[str, Tuple[int, int]]]]:
    """
    Extracts location entities from the given text using locationtagger
    and returns both the names and their character spans.

    Returns:
        {
            'countries': [{'name': ..., 'span': (...)}, ...],
            'regions': [{'name': ..., 'span': (...)}, ...],
            'cities': [{'name': ..., 'span': (...)}, ...]
        }
    """
    # Avoid misinterpreting non-sentences as URLs
    if is_likely_url(text.strip()):
        return {'countries': [], 'regions': [], 'cities': []}

    # Heuristically skip very short inputs
    if len(text.strip().split()) < 2:
        return {'countries': [], 'regions': [], 'cities': []}

    print(f"[DEBUG] Extracting locations from: {text}")
    place_entity = locationtagger.find_locations(text)

    def find_spans(locations: List[str]) -> List[Dict[str, Tuple[int, int]]]:
        results = []
        for loc in locations:
            for match in re.finditer(re.escape(loc), text, re.IGNORECASE):
                results.append({'name': loc, 'span': match.span()})
                break  # only record the first occurrence
        return results

    return {
        'countries': find_spans(place_entity.countries),
        'regions': find_spans(place_entity.regions),
        'cities': find_spans(place_entity.cities)
    }

def extract_term_years(text: str) -> Optional[float]:
    """
    Extracts the mortgage term (amortization period) from the text and normalizes it to years.
    Filters out mentions related to payment frequency.

    Looks for context like:
    - "amortization period is 25 years"
    - "loan term of 30 years"
    - "repay over 20 years"
    """

    normalized = text.lower()

    # Terms that usually signal amortization or loan duration
    anchors = [
        r"(amortization|loan term|loan duration|repay(?:ment)?(?: over)?|mortgage term|term of|for a term of)",
    ]

    # Time expression pattern
    time_expr = r"(\d+(?:\.\d+)?)\s*(year|month|week|day)s?"

    # Look for anchored term expressions
    for anchor in anchors:
        match = re.search(anchor + r".*?" + time_expr, normalized)
        if match:
            start, end = match.span()
            value = float(match.group(2))
            unit = match.group(3).lower()
            factor = {
                "year": 1, "month": 1 / 12, "week": 1 / 52, "day": 1 / 365
            }.get(unit, 0)
            return {"value": value * factor, "span": (start, end)}
    return None

def extract_payment_frequency(text: str) -> Optional[Dict[str, Union[str, tuple]]]:
    """
    Extracts the payment frequency from the text and returns the value with its character span.

    Recognizes:
    - weekly
    - bi-weekly / biweekly
    - accelerated weekly / bi-weekly
    - semi-monthly
    - bi-monthly
    - every 2 weeks
    - monthly, quarterly, annually/yearly
    """
    normalized_text = text.lower()

    patterns = [
        (r"accelerated\s+bi\s*weekly", "accelerated bi-weekly"),
        (r"accelerated\s+weekly", "accelerated weekly"),
        (r"bi\s*weekly", "bi-weekly"),
        (r"weekly", "weekly"),
        (r"semi\s*monthly", "semi-monthly"),
        (r"bi\s*monthly", "bi-monthly"),
        (r"every\s+2\s+weeks?", "bi-weekly"),
        (r"monthly", "monthly"),
        (r"quarterly", "quarterly"),
        (r"(annually|yearly)", "annually"),
    ]

    for pattern, label in patterns:
        match = re.search(pattern, normalized_text)
        if match:
            return {"value": label, "span": match.span()}

    return None

#def extract_interest_rate(text: str) -> Optional[float]:
#    """
#    Extracts the interest rate from the text.
#    """
#    match = re.search(r"(\d+(?:\.\d+)?)\s*%?", text)
#    if match:
#        return float(match.group(1))
#    return None

def extract_first_time_home_buyer(text: str) -> Optional[Dict[str, Union[bool, tuple]]]:
    """
    Attempts to determine if the user is a first-time home buyer.

    Returns:
        {
            "value": True or False,
            "span": (start_index, end_index)
        }
        or None if status is ambiguous or not mentioned
    """
    normalized = text.lower()

    # Clear positives
    positive_patterns = [
        r"\b(first[-\s]?time)\b.*\b(home[-\s]?buyer|buyer)\b",
        r"\bnever\b.*\b(bought|owned|purchased)\b.*\b(home|house|property)\b",
        r"\bi[' ]?m\b.*\bnew to (homeownership|buying a house|real estate)\b",
    ]

    # Clear negatives
    negative_patterns = [
        r"\b(not|n't|never)\b.*\b(first[-\s]?time)\b.*\b(home[-\s]?buyer|buyer)\b",
        r"\b(already|previously)\b.*\b(bought|owned)\b.*\b(home|house|property)\b",
        r"\bi (have|had|bought|own)\b.*\b(home|house|property)\b",
    ]

    for pattern in positive_patterns:
        match = re.search(pattern, normalized)
        if match:
            return {"value": True, "span": match.span()}

    for pattern in negative_patterns:
        match = re.search(pattern, normalized)
        if match:
            return {"value": False, "span": match.span()}

    return None

def extract_down_payment(text: str) -> Optional[Dict[str, Union[str, float, tuple]]]:
    """
    Attempts to extract a down payment expressed as either an absolute dollar amount or a percentage.

    Returns:
        {
            "type": "absolute" or "percent",
            "value": float,
            "span": (start_index, end_index)
        }
        or None if not detected
    """
    normalized = text.lower()

    # Match absolute dollar amounts
    abs_match = re.search(
        r"(?:down\s*payment|putting\s*down|i\s*have|saved\s*up|put\s*down).*?\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+)", 
        normalized
    )
    if abs_match:
        value = float(abs_match.group(1).replace(",", ""))
        span = abs_match.span()
        return {"type": "absolute", "value": value, "span": span}

    # Match percentages (e.g., 5%, 20 %)
    percent_match = re.search(
        r"(?:down\s*payment|putting\s*down|planning|saving\s*up)?.*?(\d{1,2}(?:\.\d+)?)\s*%", 
        normalized
    )
    if percent_match:
        percent = float(percent_match.group(1))
        span = percent_match.span()
        return {"type": "percent", "value": percent, "span": span}

    return None

def is_overlapping(span, used_spans):
    return any(not (span[1] <= s[0] or span[0] >= s[1]) for s in used_spans)

def extract_mortgage_parameters(text: str, fsm) -> Dict[str, Optional[float]]:
    """
    Extracts mortgage parameters from natural language text.
    """
    params = {}
    feedback_messages = []
    used_spans = []

    try:
        price_result = extract_price(text)
        if price_result is not None:
            span = price_result["span"]
            if not is_overlapping(span, used_spans):
                used_spans.append(span)            
                params["price"] = price_result["value"]
                fsm.receive_input('price', price_result["value"])

        locations_result = extract_locations(text)
        if locations_result is not None:
            # Flatten the dictionary to get all locations
            all_locations = []
            for loc_type, loc_list in locations_result.items():
                for loc in loc_list:
                    all_locations.append(loc)
            # Sort by span length to avoid overlapping
            all_locations.sort(key=lambda x: x["span"][1] - x["span"][0])
            # Check for overlapping spans
            for loc in all_locations:
                span = loc["span"]
                if not is_overlapping(span, used_spans):
                    used_spans.append(span)
                    params["location"] = loc["name"]
                    fsm.receive_input('location', loc["name"])
                    break

        term_years_result = extract_term_years(text)
        if term_years_result is not None:
            span = term_years_result["span"]
            if not is_overlapping(span, used_spans):
                used_spans.append(span)            
                params["term_years"] = term_years_result["value"]
                fsm.receive_input('term_years', term_years_result["value"])

        payment_frequency_result = extract_payment_frequency(text)
        if payment_frequency_result is not None:
            span = payment_frequency_result["span"]
            if not is_overlapping(span, used_spans):
                used_spans.append(span)            
                params["payment_frequency"] = payment_frequency_result["value"]
                fsm.receive_input('payment_frequency', payment_frequency_result["value"])

        first_time_home_buyer_result = extract_first_time_home_buyer(text)
        if first_time_home_buyer_result is not None:
            span = first_time_home_buyer_result["span"]
            if not is_overlapping(span, used_spans):
                used_spans.append(span)            
                params["first_time_home_buyer"] = first_time_home_buyer_result["value"]
                fsm.receive_input('first_time_home_buyer', first_time_home_buyer_result["value"])

        down_payment_result = extract_down_payment(text)
        if down_payment_result is not None and "price" in params:
            span = down_payment_result["span"]
            if not is_overlapping(span, used_spans):
                used_spans.append(span)
                min_down_payment = calculate_min_downpayment(params["price"])
                if (down_payment_result["value"])["type"] == "absolute":
                    down_payment = (down_payment_result["value"])["value"]
                    if down_payment < min_down_payment:
                        feedback_messages.append(f"Down payment of ${down_payment} is less than the minimum required ${min_down_payment}.")
                    params["down_payment"] = down_payment
                    fsm.receive_input('down_payment', down_payment)
                elif (down_payment_result["value"])["type"] == "percent":
                    down_payment = ((down_payment_result["value"])["value"] / 100) * params["price"]
                    if down_payment < min_down_payment:
                        feedback_messages.append(f"Down payment of ${down_payment} is less than the minimum required ${min_down_payment}.")
                    params["down_payment"] = down_payment
                    fsm.receive_input('down_payment', down_payment)

        #interest_rate = extract_interest_rate(text)
        #if interest_rate is not None:
        #    params["interest_rate"] = interest_rate
        #    fsm.receive_input('interest_rate', interest_rate)

    except Exception as e:
        print(f"[WARNING] Parameter extraction failed: {e}")

    return feedback_messages

def calculate_mortgage_payment(price: float, down_payment_percent: float, insured: bool=True, payment_frequency: str="monthly", term_years: int=25, first_time_home_buyer: bool=False) -> str:
    """Calculates the mortgage payment based on payment frequency."""

    print("calculate_mortgage_payment()")
    print("price:", price)
    print("down_payment_percent:", down_payment_percent)
    print("insured:", insured)
    print("payment_frequency:", payment_frequency)
    print("term_years:", term_years)
    print("first_time_home_buyer:", first_time_home_buyer)
    notes = []
    try:
        if first_time_home_buyer:
            first_time_home_buyer_msg = "for a first time home buyer, "
        else:
            first_time_home_buyer_msg = ""
        
        if price <= 0:
            notes.append("The property price must be greater than zero.")

        if down_payment_percent < 5 or down_payment_percent > 100:
            down_payment_percent_str = f"{down_payment_percent:.2f}"
            notes.append(f"Down payment percentage must be between 0 and 100. Provided: {down_payment_percent_str}")
            down_payment_percent = max(0, min(down_payment_percent, 100))

        original_down_payment_percent = down_payment_percent
        min_down_payment = calculate_min_downpayment(price)
        min_down_payment_percent = (min_down_payment / price) * 100.0
        min_down_payment_percent = round(min_down_payment_percent, 2)
        down_payment = (down_payment_percent / 100.0) * price
        original_down_payment = down_payment
        if down_payment < min_down_payment:
            # clamp to minimum down payment
            down_payment = min_down_payment
            down_payment_percent = (down_payment / price) * 100.0
            down_payment_percent = round(down_payment_percent, 2)
            min_down_payment_str = f"{min_down_payment:.2f}"
            min_down_payment_percent_str = f"{min_down_payment_percent:.2f}"
            original_down_payment_str = f"{original_down_payment:.2f}"
            original_down_payment_percent_str = f"{original_down_payment_percent:.2f}"
            notes.append(f"Down payment of ${original_down_payment_str} ({original_down_payment_percent_str}%) is less than the minimum required ${min_down_payment_str} ({min_down_payment_percent_str}%). Using minimum down payment instead.")
        print("down_payment:", down_payment)
        print("down_payment_percent:", down_payment_percent)

        if down_payment > price:
            # clamp to maximum down payment
            down_payment = price
            down_payment_percent = 100.0
            original_down_payment_str = f"{original_down_payment:.2f}"
            notes.append(f"Down payment of ${original_down_payment} ({original_down_payment_str}%) exceeds the property price ${price}. Using maximum down payment instead.")
        print("clamped down_payment:", down_payment)

        insurance = calculate_cmhc_insurance(price, down_payment)
        if price >= 1500000:
            notes.append("For properties priced at $1,500,000 or above, the mortgage insurance is not applicable as the minimum down payment is 20%.")
        print("insurance:", insurance)

        principal = price - down_payment + insurance
        if principal <= 0:
            notes.append("The calculated principal is less than or equal to zero, which is not valid for a mortgage.")
        print("principal:", principal)

        if first_time_home_buyer:
            term_years = 30
            notes.append("For first time home buyers, the term is set to 30 years by default.")

        interest_rate = get_interest_rate_percent(term_years, insured)
        if interest_rate is None:
            notes.append("Interest rate could not be determined. Please check the mortgage terms or contact your lender.")
            return "Error: " + ", ".join(notes)
        print("interest_rate:", interest_rate)
        annual_rate_fraction = float(interest_rate)/100.0
        r_annual = float(annual_rate_fraction)

        # Frequency settings
        frequency_map = {
            "monthly": 12,
            "bi-weekly": 26,
            "accelerated bi-weekly": 26,
            "weekly": 52,
            "accelerated weekly": 52,
            "semi-monthly": 24,
            "bi-monthly": 6,
            "quarterly": 4,
            "annually": 1
        }

        freq = frequency_map.get(payment_frequency.lower(), 12)
        print("freq:", freq)
        print(type(r_annual))
        print(type(freq))
        r = r_annual / freq  # interest per period
        print("r:", r)
        print(type(term_years))
        n = term_years * freq  # number of payments
        print("n:", n)

        print(type(principal))
        if r == 0:
            payment = principal / n
        else:
            payment = principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

        down_payment_str = f"{down_payment:.2f}"
        down_payment_percent_str = f"{down_payment_percent:.2f}"
        insurance_str = f"{insurance:.2f}"
        payment_str = f"{payment:.2f}"
        msg = f"For a property with market price ${price}, {first_time_home_buyer_msg} with ${down_payment_str} ({down_payment_percent_str}%) down payment, and mortgage insurance ${insurance_str}, the estimated {payment_frequency} payment for a mortgage of term {term_years} years is ${payment_str}. "
        notes_str = " ".join(notes)
        if notes_str:
            return f"{msg} Please note: {notes_str}"
        else:
            return msg

    except Exception as e:
        return f"Calculation error: {e}"


context_need_price = "Assume that user still has to provide the price of the property they are interested in. "

context_need_location = "Assume that user still has to provide the location of the property (nearest city or town). "

system_context_result = "Notify the user concisely of the result of the calculations. "

user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'

def try_extract_json_array(text):
    try:
        # Try direct JSON parse first
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback to extracting JSON array inside larger text
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None

#conversation = [
#    {"role": "system", "content": system_prompt},
#    {"role": "user", "content": user_prompt_1},
#    {"role": "assistant", "content": assistant_response_1},
#    {"role": "user", "content": user_prompt_2},
#]

# flatten the list of conversation turns before inputting to model.generate()
def format_conversation(conversation) -> str:
    prompt = ""
    for turn in conversation:
        role = turn["role"]
        content = turn["content"].strip()
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
    prompt += "<|assistant|>\n"  # Signal to model to generate next assistant message
    return prompt

# Tools should be a list of functions stored in JSON format
tools = [
    {
        "name": "calculate_mortgage_payment",
        "description": "Calculates periodic mortgage payments using the following 6 parameters: price of the property (required), down payment as a percentage of the price (required), if mortgage is insured or not (optional), frequency of payments (optional), amortization period (optional), and if they are a first time home buyer (optional). Interest rates are looked up by the tool internally.",
        "parameters": {
            "price": {
                "description": "The price of the property in Canadian dollars",
                "type": "float"
            },
            "down_payment_percent": {
                "description": "The down payment as a percentage of purchase price of the property.",
                "type": "float"
            },
            "insured": {
                "description": "Whether the mortgage is insured or not",
                "type": "boolean",
                "default": True
            },
            "payment_frequency": {
                "description": "The frequency of payments (e.g., monthly, bi-weekly, weekly)",
                "type": "str",
                "default": "monthly"
            },
            "term_years": {
                "description": "The amortization period of the mortgage",
                "type": "int",
                "default": 25
            },
            "first_time_home_buyer": {
                "description": "Whether the user is a first-time home buyer",
                "type": "boolean",
                "default": False
            }
        }
    },
]

tool_description = """
Tool: calculate_mortgage_payment
Description: Calculates periodic mortgage payments using the following parameters.
Required:
  - price: float
Optional:
  - down_payment_percent: float = 5.0
  - insured: boolean = True
  - payment_frequency: str = "monthly"
  - term_years: int = 25
  - first_time_home_buyer: boolean = False
"""

def generate_response(messages: List[Dict[str, Any]]) -> str:
    """Generates the initial text response from the LLM."""
    global model, tokenizer, generation_config

    #messages = [
    #    {
    #        "role": "system",
    #        "content": "You are a helpful professional assistant for real estate agent Susan Varty's potential clients located in Ontario, Canada. ",
    #        "tools": f"{json.dumps(tools)}"
    #    },
    #    {
    #        "role": "user",
    #        "content": ""
    #    }
    #]

    # Step 1: Start with base system prompt
    #system_prompt = system_context or ""
    #additional_instructions = []

    # Step 2: Dynamically add more instructions as needed

    #feedback_messages = extract_mortgage_parameters(user_input, fsm)

    # select the message to add to general context based on what information it still needs in order to calculate mortgage
    #print(f"[DEBUG]Still needed: {fsm.missing_fields()}")
    #missing = fsm.missing_fields()
    #if fsm.state == "estimate":
        # do calculation, and add result to system context to report result to user
    #    calculation_result = calculate_mortgage_payment(fsm)
    #    additional_instructions.append(system_context_result)
    #    additional_instructions.append(calculation_result)
    #    if feedback_messages:
    #        additional_instructions.extend(msg.strip() for msg in feedback_messages)
    #elif fsm.state == "collecting_location":
        #prompts_for_missing = [prompt_parts[k] for k in missing if k in prompt_parts]
    #    additional_instructions.append(context_need_location)
    #    if feedback_messages:
    #        additional_instructions.extend(msg.strip() for msg in feedback_messages)
    #elif fsm.state == "collecting_price":
    #    #prompts_for_missing = [prompt_parts[k] for k in missing if k in prompt_parts]
    #    additional_instructions.append(context_need_price)
    #    if feedback_messages:
    #        additional_instructions.extend(msg.strip() for msg in feedback_messages)
    #elif fsm.state == "book_meeting":
    #    print(f"[DEBUG] Book Meeting hook triggered")
    #elif fsm.state == "show_listings":
    #    print(f"[DEBUG] Show Listings hook triggered")
    #elif fsm.state == "error":
    #    if feedback_messages:
    #        additional_instructions.extend(msg.strip() for msg in feedback_messages)
    
    # Step 3: Merge them into system_prompt
    #if additional_instructions:
    #    system_prompt += " " + " ".join(additional_instructions)

    # Step 4: Construct chat-style prompt with special tokens
    #prompt = (
    #    f"<|system|>{system_prompt.strip()}<|end|>\n"
    #    f"<|user|>{user_input.strip()}\n"
    #    "<|end|><|assistant|>\n"
    #)

    #print(f"[DEBUG] Calling model pipeline with:\n{prompt}")

    try:
        #messages[1]["content"] = user_input.strip()
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        #formatted_prompt = format_conversation(conversation)

        #inputs = processor(
        #    flattened_prompt,
        #    images=None,
        #    return_tensors='pt'
        #).to("cuda")

        # for debugging
        for key, value in inputs.items():
            if hasattr(value, "shape"):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: {value}")

        # for debugging
        input_ids = inputs["input_ids"]
        decoded_prompt = processor.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        print("Decoded prompt:\n", decoded_prompt[0])

        #class InputMode(Enum):
        #    LANGUAGE = 0
        #    VISION = 1
        #    SPEECH = 2
        #    VISION_SPEECH = 3

        output = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            generation_config=generation_config,
            input_mode=0,
            do_sample=True,
        )
        print(tokenizer.decode(output[0][len(inputs['input_ids'][0]):]))

        tokenizer.batch_decode(output)

        response = tokenizer.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(f"[DEBUG] Model response: {response}")

        tool_call_id = ''.join(random.choices(string.ascii_letters + string.digits, k=9))

        #tool_call_list = try_extract_json_array(response)
        #if tool_call_list:
        # we are searching for a reference to a tool call in the response
        try:
            tool_call = json.loads(response)[0]
        except (json.JSONDecodeError, IndexError):
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                json_part = json_match.group(0)
                try:
                    tool_call = json.loads(json_part)[0]
                except json.JSONDecodeError:
                    print("[DEBUG] Failed to parse JSON from extracted text.")
                    tool_call = None
            else:
                print("[DEBUG] No JSON pattern found in response.")
                tool_call = None

        if tool_call and isinstance(tool_call, dict) and "name" in tool_call and "arguments" in tool_call:
            print(f"[DEBUG] Tool call detected: {tool_call}")
        else:
            print("[DEBUG] No valid tool call detected in response.")
            messages.append({"role": "assistant", "content": response.strip()})
            return messages

        # proceed to handle the tool call...
        function_name = tool_call["name"]
        arguments = tool_call["arguments"]

        #messages.append({"role": "assistant", "tool_calls": [{"type": "function", "id": tool_call_id, "function": function_name}]})

        # remember result of tool call in the conversation history to feed into next iteration of chat
        # note that role "tool" is for OpenAI models only
        #result = globals()[func_name](3, 5) if func_name is in global scope
        result = globals()[function_name](**arguments)

        #messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": "calculate_mortgage_payment", "content": str(result)})
        print(messages)

        response = result

        messages.append({"role": "assistant", "content": response.strip()})

        #    except Exception as tool_exception:
        #        response = f"Sorry, an internal error occurred while calculating your result: {tool_exception}"

            #conversation.append({"role": "assistant", "content": f"Your estimated payment is {tool_result}"})
            
            #messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": "calculate_mortgage_payment", "content": str(tool_result)})

            #print(messages)
            #print(response)
        #    return response
        #else:
        #    print("[DEBUG] Tool was not called.")
            # fallback to just returning the text as-is
        return messages
    except Exception as e:
        traceback.print_exc()
        print(f"[ERROR] Error during generation: {e}")  # Log the detailed error
        messages.append({"role": "assistant", "content": "Sorry, I encountered an error while generating a response."})
        return messages

system_prompt = (
    "You are a helpful, conversational assistant supporting real estate agent Susan Varty. "
    "Your role is to help potential clients in Ontario, Canada estimate their mortgage payments. "
    "You have access to a tool called `calculate_mortgage_payment`, which you should call directly by emitting a JSON block with the required parameters anytime you have enough information to perform a mortgage calculation.\n\n"

    "You must respond in a friendly, helpful, and conversational tone—do not explain that you are an AI or describe hypothetical interactions. "
    "Instead, respond directly and naturally to the user, calling the tool when appropriate.\n\n"

    "**Tool Description**:\n"
    "- name: calculate_mortgage_payment\n"
    "- description: Calculates periodic mortgage payments using the provided parameters.\n"
    "- required parameter: `price` (float)\n"
    "- optional parameters: `down_payment_percent` (float), `insured` (bool), `payment_frequency` (str), `term_years` (int), `first_time_home_buyer` (bool)\n\n"

    "When you have enough information, call the tool like this:\n"
    "```json\n"
    "{\n"
    "  \"price\": 300000,\n"
    "  \"down_payment_percent\": 20.0,\n"
    "  \"insured\": true,\n"
    "  \"payment_frequency\": \"monthly\",\n"
    "  \"term_years\": 25,\n"
    "  \"first_time_home_buyer\": false\n"
    "}\n"
    "```\n"
    "Then wait for the response and provide the result to the user."
)

@app.post("/chat")
async def chat_endpoint(request: Request):
    global model_ready, model, tokenizer, tool_description, system_prompt
    
    cleanup_inactive_sessions()

    if not model_ready or model is None or tokenizer is None:
        return JSONResponse({"error": "Model not ready"}, status_code=503)

    req_json = await request.json()
    # session id is generated by frontend javascript running in remote browser
    session_id = req_json.get("session_id")
    user_input = req_json.get("prompt")

    print(f"[DEBUG] Session ID: {session_id} | Prompt: {user_input}")

    if not session_id:
        return JSONResponse({"error": "Missing session_id"}, status_code=400)

    with session_lock:
        session = sessions.setdefault(session_id, {
            "history": [],
            "last_updated": time.time()
        })
        session["last_updated"] = time.time()

    messages = session["history"]
    if not messages:
        messages = [
            {"role": "system", 
             "content": "You are a helpful, conversational assistant supporting real estate agent Susan Varty. "
             "Your role is to help potential clients in Ontario, Canada estimate their mortgage payments. "
             "You have access to a tool called `calculate_mortgage_payment`, which you should call directly by emitting a JSON block with the required parameters anytime you have at least both the price of the property they are interested in and their down payment as a percentage of the price.",
             "tools": f"{json.dumps(tools)}"
            },
            {"role": "user", 
             "content": user_input.strip()}
            ]
        with session_lock:
            session["history"] = messages
    else:
        with session_lock:
            messages.append({"role": "user", "content": user_input.strip()})

    print(f"[DEBUG]: {sessions}")
    #print(f"[DEBUG] Session ID: {session_id} | Active sessions: {len(conversation_state)}")
    #     
    #if not session_id:
    #    raise HTTPException(status_code=400, detail="Missing session_id")
    #if not session_id:
    #    session_id = str(uuid4())
    #with conversation_lock:
    #    session = conversation_state.setdefault(session_id, {
    #        "history": [],  # a list of dictionaries
    #        "last_updated": time.time(),
    #        "fsm": MortgageConversation()
    #    })
    #user_message = req_json.get("prompt", "").strip()
    #print(f"[DEBUG] Session ID: {session_id} | Prompt: {user_input.strip()}")

    #print(f"[DEBUG] Session exists? {'yes' if session_id in conversation_state else 'no'}")

    #print(f"[DEBUG] Session ID: {session_id}")
    #print(f"[DEBUG] Session ID: {session_id} | Active sessions: {len(conversation_state)}")
    #print(f"[DEBUG] Session ID: {session_id} | Last updated: {conversation_state[session_id]['last_updated']}")
    
    #print("[DEBUG] Tokenizer config:", tokenizer.special_tokens_map, tokenizer.padding_side, tokenizer.truncation_side)

    #print(f"[DEBUG] Entering /chat, pipe is: {pipe}") 
    #if pipe is None:
    #    pipe = TextGenerationPipeline(model=model, trust_remote_code=True)
        #print(f"[DEBUG] Initialized pipe: {pipe}, pipe.device: {getattr(pipe, 'device', None)}")
        #print("[DEBUG] Pipe tokenizer class:", type(pipe.tokenizer))
    #print(f"[DEBUG] After initialization check, pipe is: {pipe}, pipe.device: {getattr(pipe, 'device', None)}")

    #req_json = await request.json()
    #user_message = req_json.get('prompt', '').strip()
    #user_message = user_message.strip()  # Remove leading/trailing whitespace
    #user_message = re.sub(r'[\r\n]+', ' ', user_message)  # Normalize newlines
    #print(f"[DEBUG] Received prompt: {user_message}")
    #if not user_message:
    #    return JSONResponse(
    #        status_code=400,
    #        content={"error": "Missing 'prompt' in request body."}
    #    )    
    #if not isinstance(user_message, str) or not user_message:
    #    print(f"[ERROR] Invalid user message: {repr(user_message)}")
    #    return JSONResponse(
    #        status_code=400,
    #        content={"error": "Invalid prompt received. Must be non-empty string."}
    #    )

    #print(f"[DEBUG] Calling model pipeline with: {repr(user_message)}")

    # Update session with new message
    #session["history"].append({"user": user_message})
    #session["last_updated"] = time.time()

    #formatted_prompt = format_conversation(messages)
    #print(f"[DEBUG] Calling model pipeline with: '{user_input.strip()}'")

    # Generate AI response
    try:
        #stop_sequences = ["\n\n", "</s>", "<|endoftext|>"]
        #encoded_stop_sequences = []
        #for seq in stop_sequences:
        #    try:
        #        encoded = tokenizer.encode(seq, add_special_tokens=False)
        #        encoded_stop_sequences.append(encoded)
        #        print(f"[DEBUG] Encoded '{seq}': {encoded}")
        #    except Exception as e:
        #        print(f"[ERROR] Encoding '{seq}': {e}")
        #print("[DEBUG] All encoded stop sequences:", encoded_stop_sequences)
        #stop_sequence_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]
        #print("[DEBUG] Stop sequence ids:", stop_sequence_ids)
        #print("[DEBUG] eos_token_id:", tokenizer.eos_token_id)

        #fsm = session["fsm"]
        messages = generate_response(messages)  # Generate initial response
        print(f"[DEBUG] Raw model output: {messages}")  # Log the raw output

        with session_lock:
            session["history"] = messages

        ai_response = ""
        if messages and messages[-1].get("role") == "assistant":
            ai_response = messages[-1]["content"].strip()
        else:
            ai_response = "Sorry, I encountered an error while generating a response."

        #session["history"].append({"bot": ai_response})
        return JSONResponse(
            status_code=200,
            content={"response": ai_response}
        )
    except Exception as e:
        print(f"[ERROR] Model generation failed: {str(e)}")
        print(traceback.print_exc())  # Log the full traceback        
        return JSONResponse(
            status_code=500,
            content={ai_response}
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
