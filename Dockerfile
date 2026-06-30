# ── AM RAG Document QA — production image ───────────────────────────────────────
# Base: slim Python 3.11 (matches the dev environment).
FROM python:3.11-slim

# Faster, cleaner Python in containers.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Pre-cache the embedding model into the image layer dir (faster cold start).
    HF_HOME=/home/appuser/.cache/huggingface

# Minimal system deps. faiss-cpu / torch ship manylinux wheels, so no compiler needed.
# curl is used by the container HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching). Force CPU-only torch to avoid the huge
# CUDA wheels — the app runs embeddings on CPU.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# App code (runtime data dirs are gitignored / .dockerignored and created at startup).
COPY . .

# Run as a non-root user.
RUN useradd -m appuser && chown -R appuser:appuser /app /home/appuser
USER appuser

EXPOSE 8501

# App Runner / orchestrators probe this; Streamlit exposes a native health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Bind 0.0.0.0, headless, and disable XSRF/CORS so it works behind App Runner / HF proxies.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
