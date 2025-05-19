# syntax=docker/dockerfile:1.4

FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/grahamdwall/phi2-finetune"

ENV PYTHONUNBUFFERED=1d
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8
ENV HF_HOME=/workspace/data/models
ENV TRANSFORMERS_VERBOSITY=debug
# Set NLTK data path for all Python scripts in the container
ENV NLTK_DATA=/root/nltk_data

# Install Python build dependencies + git + lib dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    gcc \
    wget \
    sudo \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only version
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

# Install core libraries first
# Assuming that RunPod has GPU drivers installed. pip install torch was bloating the image by 9GB
COPY requirements.txt .

# Install torch with CUDA 12.1 from the custom PyTorch index
RUN pip install --no-cache-dir torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# Then install the rest of the requirements from your requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy model (en_core_web_sm)
#RUN mkdir -p ./spacy
#ENV SPACY_DATA_PATH="./spacy"
RUN python -m spacy download en_core_web_sm

# Download NLTK data programmatically
RUN python -m nltk.downloader punkt averaged_perceptron_tagger maxent_ne_chunker words treebank maxent_treebank_pos_tagger

# Add cloudflared
RUN wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && \
    dpkg -i cloudflared-linux-amd64.deb && \
    rm cloudflared-linux-amd64.deb

# Install accelerate CLI (optional: configure later)
RUN accelerate config default

# Copy your app code
COPY train_phi2_lora.py .
COPY evaluate_phi2.py .
COPY train_server.py .
COPY mortgage_convo_fsm.py .
COPY mortgage_rates.py .
COPY start.sh .
RUN mkdir -p ./data/test
COPY mortgage_finetune_1000.jsonl ./data/test/

# --- Copy Cloudflare credentials (during docker build or mount at runtime) ---
# Make sure you COPY cert.pem and tunnel credentials json into image or volume
COPY cloudflared /root/.cloudflared

# --- Set up sysctl for large socket buffers ---
RUN sysctl -w net.core.rmem_max=2500000 || true

# Copy your baked-in LLM models
#COPY phi2_model_full/ ./models/microsoft/phi-2

# test spacy so that it won't fail at runtime
#RUN python -c "import spacy; spacy.load('en_core_web_sm')"

EXPOSE 8000

# --- Start script ---
RUN chmod +x start.sh
CMD ["bash", "start.sh"]

# for testing
#CMD ["uvicorn", "phi2_api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
# for production
#CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "phi2_finetune:app", "--bind", "0.0.0.0:8080"]
#CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "train_server:app", "--bind", "0.0.0.0:8080"]
