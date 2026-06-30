"""
config.py - Central configuration for the RAG application.
All tunable parameters live here; nothing is hardcoded elsewhere.
"""

import os
from dotenv import load_dotenv

# Load .env so secrets are available as environment variables
load_dotenv()

# --- Paths ---
UPLOAD_FOLDER: str = "uploads"
VECTOR_STORE_PATH: str = "vector_store"   # PageIndex / fixed strategy
STORAGE_PATH: str = "storage"             # Parent-Child strategy

# --- Chunking (PageIndex / fixed strategy) ---
CHUNK_SIZE: int = 1500          # ~50% bigger → fewer mid-list split boundaries
CHUNK_OVERLAP: int = 200

# --- Chunking (Parent-Child strategy) ---
PARENT_CHUNK_SIZE: int = 6000    # ~1 500 tokens — full context block for LLM
CHILD_CHUNK_SIZE: int = 1000     # ~256 tokens — precise retrieval unit
CHILD_CHUNK_OVERLAP: int = 200   # ~50 tokens

# --- Embedding model ---
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# --- LLM ---
# Options: "deepseek" | "ollama" | "huggingface"
LLM_PROVIDER: str = "deepseek"

# DeepSeek (primary)
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL: str = "deepseek-v4-flash"

# Ollama (local alternative)
OLLAMA_MODEL: str = "llama3.2"
OLLAMA_BASE_URL: str = "http://localhost:11434"

# HuggingFace (fallback, no API key needed)
HF_MODEL: str = "google/flan-t5-base"

# --- Demo rate limit ---
MAX_QUESTIONS_PER_SESSION: int = 12   # per browser session; resets on refresh

# --- Retrieval ---
TOP_K: int = 5                  # default fallback
PAGE_INDEX_TOP_K: int = 7       # more small chunks → ~10 500 chars LLM context
PARENT_CHILD_TOP_K: int = 3     # fewer large parent blocks → ~18 000 chars LLM context

# --- LangSmith tracing (optional) ---
LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "simple-doc-rag")
LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# Activate LangSmith tracing if API key is provided
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGSMITH_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = LANGCHAIN_ENDPOINT

# --- Ensure required directories exist ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(STORAGE_PATH, exist_ok=True)
