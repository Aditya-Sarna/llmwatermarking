# HF Spaces Docker — FastAPI backend for LLM Watermark Studio
FROM python:3.11-slim

# HF Spaces runs as a non-root user (uid 1000)
RUN useradd -m -u 1000 user
WORKDIR /app

# System deps for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller image), then the rest
RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HF Spaces: serve on port 7860 and cache models to /tmp (writable)
ENV PORT=7860
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache

USER user

# CORS_ORIGINS should be set as a Space secret to your Netlify URL
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
