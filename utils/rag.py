"""
rag.py — Public API for the RAG pipeline.

Delegates to the LangGraph-powered agentic graph in utils/graph.py.
LangSmith traces all graph nodes automatically via LCEL callbacks.

Both PageIndex and Parent-Child strategies flow through the same graph;
the retrieve node dispatches internally based on doc_strategy.
"""

import time
from typing import Dict, Any, Generator

from utils.graph import get_rag_graph
from utils.llm import get_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


_ANSWER_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant that answers questions strictly based on the provided context.

RULES:
- Use ONLY the information in the context below.
- If the answer is not found in the context, respond EXACTLY with:
  "I could not find this information in the selected document."
- Do not add information from your training data.
- Be concise and accurate.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
)


def answer_question(
    doc_id: str,
    question: str,
    session_id: str = "default",
    doc_strategy: str = "page_index",
) -> Dict[str, Any]:
    """
    Run the agentic RAG graph and return the answer + sources.
    LangGraph + LCEL handle LangSmith tracing automatically.
    """
    graph = get_rag_graph()
    state = graph.invoke({
        "doc_id": doc_id,
        "session_id": session_id,
        "doc_strategy": doc_strategy,
        "question": question,
        "rewritten_question": "",
        "chunks": [],
        "answer": "",
        "retry_count": 0,
    })

    answer = state.get("answer", "")
    if not answer or answer == "__no_chunks__":
        answer = "I could not find this information in the selected document."

    return {
        "answer": answer,
        "sources": state.get("chunks", []),
        "rewritten_question": state.get("rewritten_question", ""),
    }


def stream_answer(
    doc_id: str,
    question: str,
    session_id: str = "default",
    doc_strategy: str = "page_index",
) -> tuple[Generator, list, dict]:
    """
    Run retrieval + grading via the graph, then stream the LLM answer token by token.

    Returns:
        (token_stream_generator, sources_list, meta_dict)

    meta_dict keys:
        retry_count        — how many times the query was rewritten (0 = no rewrite)
        rewritten_question — the rewritten query string (empty if no rewrite)
        retrieval_ms       — milliseconds spent on retrieve + grade + optional rewrite
    """
    from utils.graph import retrieve, grade_chunks, rewrite_query, RAGState

    t0 = time.time()

    state: RAGState = {
        "doc_id": doc_id,
        "session_id": session_id,
        "doc_strategy": doc_strategy,
        "question": question,
        "rewritten_question": "",
        "chunks": [],
        "answer": "",
        "retry_count": 0,
    }

    state = retrieve(state)

    if not state["chunks"]:
        meta = {"retry_count": 0, "rewritten_question": "", "retrieval_ms": int((time.time() - t0) * 1000)}
        def _empty():
            yield "I could not find this information in the selected document."
        return _empty(), [], meta

    state = grade_chunks(state)

    if state.get("answer") == "__rewrite__":
        state = rewrite_query(state)
        state = retrieve(state)

    state["answer"] = ""

    retrieval_ms = int((time.time() - t0) * 1000)

    meta = {
        "retry_count":        state.get("retry_count", 0),
        "rewritten_question": state.get("rewritten_question", ""),
        "retrieval_ms":       retrieval_ms,
    }

    if not state["chunks"]:
        def _empty():
            yield "I could not find this information in the selected document."
        return _empty(), [], meta

    context = "\n\n".join(
        f"[Excerpt {i} – Page {c['page']}]\n{c['text']}"
        for i, c in enumerate(state["chunks"], start=1)
    )

    chain = _ANSWER_PROMPT | get_llm() | StrOutputParser()
    stream = chain.stream({"context": context, "question": question})

    return stream, state["chunks"], meta
