"""
embeddings.py — Load the sentence-transformer embedding model.

The model is loaded once and reused (module-level singleton) so we don't
re-download or re-initialise it on every request.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL

# Module-level singleton — initialised on first import
_embeddings_instance: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return (and lazily create) the shared embedding model instance.

    The model runs on CPU by default, which works everywhere.
    Embeddings are L2-normalised so cosine similarity == dot product.
    """
    global _embeddings_instance

    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embeddings_instance
