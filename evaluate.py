"""
evaluate.py — RAGAS evaluation of the existing LangGraph RAG pipelines.

This script evaluates the REAL production pipeline (same LangGraph StateGraph
that powers the Streamlit app) — not a reimplementation.

Usage:
    python evaluate.py \\
        --doc "vector vs vectorless RAG.pdf" \\
        --questions "What is Vector RAG?" "What are its limitations?" \\
        [--ground_truths "Vector RAG uses..." "Limitations include..."]

    # List all indexed documents:
    python evaluate.py --list

How it works:
    1. Reads vector_store/registry.json to find the indexed document.
    2. Calls answer_question() (the real LangGraph pipeline) for each question.
    3. Maps answer + retrieved source chunks → RAGAS SingleTurnSample objects.
    4. Evaluates with DeepSeek as LLM judge + local all-MiniLM-L6-v2 embeddings.
    5. Prints a scored table; saves results/ragas_eval.csv.

Metrics:
    faithfulness      — Answer only uses information from retrieved context.
    answer_relevancy  — Answer is directly on-topic to the question.
    context_precision — Retrieved chunks are ranked by relevance. (needs --ground_truths)
    context_recall    — Context covers the information in the reference answer. (needs --ground_truths)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()

# ── Validate API key early ─────────────────────────────────────────────────────
if not os.getenv("DEEPSEEK_API_KEY"):
    print("ERROR: DEEPSEEK_API_KEY not set. Add it to your .env file.")
    sys.exit(1)


# ── Registry helpers ───────────────────────────────────────────────────────────

def _load_registry() -> dict:
    path = os.path.join("vector_store", "registry.json")
    if not os.path.exists(path):
        print("ERROR: No registry found. Upload at least one PDF in the Streamlit app first.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _find_doc(registry: dict, name_or_id: str) -> tuple[str, dict]:
    """Return (doc_id, doc_info) matching a name substring or exact doc_id."""
    # Exact ID match first
    if name_or_id in registry:
        return name_or_id, registry[name_or_id]

    # Case-insensitive name substring match
    needle = name_or_id.lower()
    matches = [
        (doc_id, info)
        for doc_id, info in registry.items()
        if needle in info.get("name", "").lower()
    ]
    if not matches:
        print(f"ERROR: No document found matching '{name_or_id}'.")
        print("Run with --list to see available documents.")
        sys.exit(1)
    if len(matches) > 1:
        names = [info["name"] for _, info in matches]
        print(f"ERROR: Multiple documents match '{name_or_id}': {names}")
        print("Use --doc_id to specify the exact document ID.")
        sys.exit(1)
    return matches[0]


def _list_documents(registry: dict) -> None:
    rows = [
        [
            doc_id,
            info.get("name", "?"),
            info.get("chunking_strategy", "?"),
            info.get("pages", "?"),
            info.get("chunks", "?"),
        ]
        for doc_id, info in registry.items()
    ]
    print(tabulate(rows, headers=["doc_id", "name", "strategy", "pages", "chunks"], tablefmt="grid"))


# ── RAGAS setup ────────────────────────────────────────────────────────────────

def _build_ragas_config():
    """
    Build RAGAS-compatible wrappers around DeepSeek (judge) and
    local HuggingFace embeddings (all-MiniLM-L6-v2).

    Reuses the same model names / base_url as config.py — no separate config.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, EMBEDDING_MODEL

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=DEEPSEEK_MODEL,
            base_url=DEEPSEEK_BASE_URL,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.0,    # deterministic scoring
            max_tokens=1024,
        )
    )
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )
    return llm, emb


# ── Evaluation core ────────────────────────────────────────────────────────────

def _run_pipeline(
    doc_id: str,
    doc_info: dict,
    questions: list[str],
) -> list[dict]:
    """
    Run the real LangGraph pipeline on all questions.

    Returns a list of dicts: {question, answer, contexts, rewritten_question}
    """
    from utils.rag import answer_question

    strategy = doc_info.get("chunking_strategy", "page_index")
    session_id = doc_info.get("session_id", "default")

    results = []
    for i, question in enumerate(questions, start=1):
        print(f"  Q{i}: {question}")
        result = answer_question(
            doc_id=doc_id,
            question=question,
            session_id=session_id,
            doc_strategy=strategy,
        )
        # Extract chunk texts as RAGAS contexts
        contexts = [s["text"] for s in result.get("sources", []) if s.get("text")]
        rewritten = result.get("rewritten_question", "")

        suffix = f" [rewritten → '{rewritten[:60]}…']" if rewritten else ""
        print(f"     → {len(contexts)} chunks retrieved{suffix}")
        print(f"     → {result['answer'][:120]}{'…' if len(result['answer']) > 120 else ''}")

        results.append({
            "question": question,
            "answer": result["answer"],
            "contexts": contexts,
            "rewritten_question": rewritten,
        })

    return results


def _evaluate_with_ragas(
    pipeline_results: list[dict],
    ground_truths: list[str] | None,
    llm,
    emb,
) -> dict[str, float]:
    """
    Build RAGAS samples and run evaluation.

    Returns {metric_name: mean_score}.
    """
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    has_gt = bool(ground_truths)
    samples = []

    for i, row in enumerate(pipeline_results):
        gt = ground_truths[i] if has_gt and i < len(ground_truths) else None
        samples.append(
            SingleTurnSample(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=row["contexts"],
                reference=gt if gt else None,
            )
        )

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
    ]
    if has_gt:
        metrics += [ContextPrecision(llm=llm), ContextRecall(llm=llm)]

    dataset = EvaluationDataset(samples=samples)
    result = evaluate(dataset=dataset, metrics=metrics)
    return {k: float(v) for k, v in result.items()}


def _save_csv(
    pipeline_results: list[dict],
    scores: dict[str, float],
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metric_names = sorted(scores.keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer", "contexts", "rewritten_question"] + metric_names)
        for row in pipeline_results:
            writer.writerow([
                row["question"],
                row["answer"],
                " | ".join(row["contexts"]),
                row["rewritten_question"],
            ] + [f"{scores.get(m, ''):.4f}" for m in metric_names])
        # Mean row
        writer.writerow(
            ["MEAN", "", "", ""] + [f"{scores.get(m, ''):.4f}" for m in metric_names]
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the existing LangGraph RAG pipelines with RAGAS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List indexed documents
  python evaluate.py --list

  # Evaluate by document name (substring match)
  python evaluate.py \\
      --doc "vector vs vectorless" \\
      --questions "What is Vector RAG?" "What are its limitations?"

  # With ground truths (enables context_precision + context_recall)
  python evaluate.py \\
      --doc "vector vs vectorless" \\
      --questions "What is Vector RAG?" \\
      --ground_truths "Vector RAG uses embeddings and FAISS to retrieve chunks..."
        """,
    )
    parser.add_argument("--list", action="store_true", help="List all indexed documents and exit.")
    parser.add_argument("--doc", help="Document name (substring) or doc_id to evaluate.")
    parser.add_argument("--questions", nargs="+", help="Test questions.")
    parser.add_argument(
        "--ground_truths", nargs="+", default=None,
        help="Optional reference answers (enables context_precision + context_recall).",
    )
    parser.add_argument(
        "--output", default="results/ragas_eval.csv",
        help="CSV output path (default: results/ragas_eval.csv).",
    )
    args = parser.parse_args()

    registry = _load_registry()

    if args.list:
        _list_documents(registry)
        return

    if not args.doc:
        parser.error("--doc is required unless --list is specified.")
    if not args.questions:
        parser.error("--questions is required.")
    if args.ground_truths and len(args.ground_truths) != len(args.questions):
        parser.error("--ground_truths must have the same number of entries as --questions.")

    doc_id, doc_info = _find_doc(registry, args.doc)
    strategy = doc_info.get("chunking_strategy", "page_index")
    has_gt = bool(args.ground_truths)

    print("=" * 62)
    print("RAGAS Evaluation — Existing LangGraph Pipeline")
    print("=" * 62)
    print(f"Document  : {doc_info['name']}")
    print(f"doc_id    : {doc_id}")
    print(f"Strategy  : {strategy}")
    print(f"Questions : {len(args.questions)}")
    print(f"GT answers: {'yes (all 4 metrics)' if has_gt else 'no (faithfulness + answer_relevancy)'}")
    print()

    # ── Step 1: run the real LangGraph pipeline ────────────────────────────────
    print("── Pipeline answers ────────────────────────────────────────")
    pipeline_results = _run_pipeline(doc_id, doc_info, args.questions)

    # ── Step 2: RAGAS evaluation ───────────────────────────────────────────────
    print("\n── RAGAS Evaluation ────────────────────────────────────────")
    print("Loading DeepSeek judge + local MiniLM embeddings…")
    llm, emb = _build_ragas_config()
    print("Scoring…")
    scores = _evaluate_with_ragas(pipeline_results, args.ground_truths, llm, emb)

    # ── Step 3: Print table ────────────────────────────────────────────────────
    print("\n── Scores ──────────────────────────────────────────────────")
    rows = [[metric, f"{score:.4f}"] for metric, score in sorted(scores.items())]
    print(tabulate(rows, headers=["Metric", "Score"], tablefmt="grid"))

    # ── Step 4: Save CSV ───────────────────────────────────────────────────────
    _save_csv(pipeline_results, scores, args.output)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
