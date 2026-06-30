"""
ragas_scorer.py — Live RAGAS scoring for a single question/answer/context tuple.

Used by app.py when the "🔬 Live RAGAS scoring" sidebar toggle is ON.
Never raises — returns {} on any failure so the UI is never blocked.

Metrics scored:
    faithfulness      — answer contains only info from retrieved context
    answer_relevancy  — answer is directly on-topic to the question
"""

from __future__ import annotations

import os
import sys
import types
from functools import lru_cache


def _patch_missing_vertexai() -> None:
    """
    ragas hard-imports ``langchain_community.chat_models.vertexai.ChatVertexAI``
    at module load. That submodule was removed in the sunset langchain_community
    0.4.x. This app never uses Vertex AI, so we register a lightweight stub module
    to satisfy the import. Self-contained — survives .venv rebuilds, touches no
    site-packages. No-op if the real module is present or the parent is missing.
    """
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        import langchain_community.chat_models as _chat_models
    except Exception:
        return  # langchain_community absent — let ragas raise its own clear error
    if hasattr(_chat_models, "vertexai"):
        return

    stub = types.ModuleType(name)

    class ChatVertexAI:  # placeholder — never instantiated in this app
        pass

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[name] = stub
    _chat_models.vertexai = stub  # so `from ...chat_models.vertexai import X` resolves


@lru_cache(maxsize=1)
def _get_ragas_wrappers():
    """
    Build and cache DeepSeek LLM + local embedding wrappers for RAGAS.
    Called once per Streamlit session; subsequent calls hit the cache.
    """
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    from utils.embeddings import get_embeddings

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=DEEPSEEK_MODEL,
            base_url=DEEPSEEK_BASE_URL,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.0,
            max_tokens=512,
        )
    )
    emb = LangchainEmbeddingsWrapper(get_embeddings())
    return llm, emb


def score_response(question: str, answer: str, contexts: list[str]) -> dict:
    """
    Score a single RAG response with RAGAS Faithfulness + AnswerRelevancy.

    Args:
        question: The user's original question.
        answer:   The LLM-generated answer.
        contexts: List of retrieved chunk texts used to produce the answer.

    Returns:
        {"faithfulness": float, "answer_relevancy": float}  — or {} on any error.
    """
    try:
        if not answer or not contexts:
            return {"_error": "No answer or contexts to score."}

        _patch_missing_vertexai()

        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import AnswerRelevancy, Faithfulness

        llm, emb = _get_ragas_wrappers()

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        result = evaluate(
            dataset=EvaluationDataset(samples=[sample]),
            metrics=[
                Faithfulness(llm=llm),
                AnswerRelevancy(llm=llm, embeddings=emb),
            ],
        )
        # EvaluationResult (ragas 0.2.x) exposes scores via to_pandas();
        # one row per sample, one column per metric. We have a single sample.
        df = result.to_pandas()
        scores = {}
        for metric in ("faithfulness", "answer_relevancy"):
            if metric in df.columns:
                val = df[metric].mean()
                if val == val:  # filter NaN (RAGAS returns NaN on a failed metric)
                    scores[metric] = round(float(val), 3)
        return scores if scores else {"_error": "RAGAS returned empty scores."}
    except Exception as e:
        return {"_error": str(e)}
