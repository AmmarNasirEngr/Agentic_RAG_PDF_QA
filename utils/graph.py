"""
graph.py — Agentic RAG pipeline built with LangGraph.

Graph flow:
    retrieve → grade_chunks → generate          (happy path)
                           ↘ rewrite_query → retrieve (retry, max 2x)

Each node is a named step that appears as a child span in LangSmith automatically.
"""

from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from config import PAGE_INDEX_TOP_K, PARENT_CHILD_TOP_K
from utils.vector_store import get_retriever
from utils.llm import get_llm


# ── State ──────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    doc_id: str
    session_id: str          # used by Parent-Child strategy for tenant isolation
    doc_strategy: str        # "page_index" | "fixed" | "parent_child"
    question: str
    rewritten_question: str
    chunks: List[Dict[str, Any]]
    answer: str
    retry_count: int


# ── Prompts ────────────────────────────────────────────────────────────────────

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

_GRADE_PROMPT = PromptTemplate.from_template(
    """You are a grader assessing whether retrieved document chunks are relevant to a question.

Question: {question}
Retrieved chunks: {context}

Answer with a single word — 'yes' if the chunks contain useful information to answer the question, or 'no' if they are irrelevant or off-topic."""
)

_REWRITE_PROMPT = PromptTemplate.from_template(
    """You are rewriting a question to improve document retrieval results.
The original question did not return relevant chunks from the document.

Original question: {question}

Rewrite it to be more specific and search-friendly. Return ONLY the rewritten question, nothing else."""
)


# ── Nodes ──────────────────────────────────────────────────────────────────────

def retrieve(state: RAGState) -> RAGState:
    """Embed the current question and fetch top-k chunks.

    Dispatches to the correct retrieval strategy based on doc_strategy:
      • "parent_child" → TenantVectorStore.retrieve_and_package (returns parent blocks)
      • "page_index" / "fixed" → PageIndex FAISS as_retriever (returns child/section chunks)
    """
    query = state.get("rewritten_question") or state["question"]

    if state.get("doc_strategy") == "parent_child":
        from utils.tenant_vector_store import TenantVectorStore
        store = TenantVectorStore(state["session_id"])
        parents = store.retrieve_and_package(query, state["doc_id"], top_k=PARENT_CHILD_TOP_K)
        chunks = [
            {
                "page": p.page,
                "text": p.text,
                "parent_id": p.parent_id,
                "child_count": len(p.child_ids),
                "chunking_strategy": "parent_child",
                "section_title": None,
            }
            for p in parents
        ]
    else:
        retriever = get_retriever(state["doc_id"], top_k=PAGE_INDEX_TOP_K)
        docs: List[Document] = retriever.invoke(query)
        chunks = [
            {
                "page": doc.metadata.get("page", "?"),
                "chunk_id": doc.metadata.get("chunk_id", "?"),
                "text": doc.page_content,
                "section_title": doc.metadata.get("section_title"),
                "section_level": doc.metadata.get("section_level"),
                "section_page_start": doc.metadata.get("section_page_start"),
                "section_page_end": doc.metadata.get("section_page_end"),
                "chunking_strategy": doc.metadata.get("chunking_strategy", "fixed"),
            }
            for doc in docs
        ]

    return {**state, "chunks": chunks}


def grade_chunks(state: RAGState) -> RAGState:
    """LLM grades whether retrieved chunks are relevant to the question."""
    if not state["chunks"]:
        return {**state, "answer": "__no_chunks__"}

    context = "\n\n".join(c["text"] for c in state["chunks"])
    chain = _GRADE_PROMPT | get_llm() | StrOutputParser()
    grade = chain.invoke({"question": state["question"], "context": context}).strip().lower()

    if "yes" in grade:
        return {**state}
    else:
        return {**state, "answer": "__rewrite__"}


def rewrite_query(state: RAGState) -> RAGState:
    """LLM rewrites the question to improve retrieval on the next attempt."""
    chain = _REWRITE_PROMPT | get_llm() | StrOutputParser()
    rewritten = chain.invoke({"question": state["question"]}).strip()
    return {**state, "rewritten_question": rewritten, "answer": "", "retry_count": state.get("retry_count", 0) + 1}


def generate(state: RAGState) -> RAGState:
    """Build grounded prompt from chunks and generate the final answer via LLM."""
    context = "\n\n".join(
        f"[Excerpt {i} – Page {c['page']}]\n{c['text']}"
        for i, c in enumerate(state["chunks"], start=1)
    )
    chain = _ANSWER_PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state["question"]}).strip()
    return {**state, "answer": answer}


# ── Routing ────────────────────────────────────────────────────────────────────

def route_after_grade(state: RAGState) -> Literal["generate", "rewrite_query", END]:
    if state.get("answer") == "__no_chunks__":
        return END
    if state.get("answer") == "__rewrite__" and state.get("retry_count", 0) < 2:
        return "rewrite_query"
    return "generate"


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_chunks", grade_chunks)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_chunks")
    graph.add_conditional_edges("grade_chunks", route_after_grade)
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


# Module-level compiled graph (built once)
_rag_graph = None


def get_rag_graph():
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = build_graph()
    return _rag_graph
