#!/bin/bash
echo "=== config.yml contents ==="
cat /root/.cloudflared/config.yml || echo "Missing config.yml"
echo "[INFO] Starting cloudflared tunnel..."
cloudflared tunnel --config /root/.cloudflared/config.yml run &
echo "[INFO] Starting FastAPI server..."
exec gunicorn train_server:app --bind 0.0.0.0:8080 --workers 1 --worker-class uvicorn.workers.UvicornWorker
echo "[INFO] FastAPI server started."