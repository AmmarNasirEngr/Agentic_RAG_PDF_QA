"""
llm.py - LLM initialisation.

Provider is set via LLM_PROVIDER in config.py:
    "deepseek"    - DeepSeek API (OpenAI-compatible), primary
    "ollama"      - Local Ollama server
    "huggingface" - Local HuggingFace pipeline (no API key, heavy download)

Instance is cached at module level to avoid reloading on every query.
"""

from __future__ import annotations
from typing import Any

from config import (
    LLM_PROVIDER,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    HF_MODEL,
)

_llm_instance: Any = None


def get_llm() -> Any:
    """Return (and lazily create) the shared LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        if LLM_PROVIDER == "deepseek":
            _llm_instance = _load_deepseek()
        elif LLM_PROVIDER == "ollama":
            _llm_instance = _load_ollama()
        else:
            _llm_instance = _load_huggingface()
    return _llm_instance


def _load_deepseek() -> Any:
    """Connect to DeepSeek via their OpenAI-compatible REST API."""
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. Add it to your .env file."
        )
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        temperature=0.1,
        max_tokens=1024,
    )
    print(f"[llm] DeepSeek ready - model: {DEEPSEEK_MODEL}")
    return llm


def _load_ollama() -> Any:
    """Connect to a locally running Ollama server."""
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            num_predict=512,
        )
        llm.invoke("hi")
        return llm
    except Exception as exc:
        print(f"[llm] Ollama unavailable ({exc}). Falling back to HuggingFace.")
        return _load_huggingface()


def _load_huggingface() -> Any:
    """Load a local HuggingFace model (no API key, ~900 MB download)."""
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    from langchain_huggingface import HuggingFacePipeline
    print(f"[llm] Loading HuggingFace model: {HF_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
    )
    return HuggingFacePipeline(pipeline=pipe)
