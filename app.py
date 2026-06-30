"""
app.py — Streamlit frontend for AM RAG Document QA.

Run with:
    streamlit run app.py
"""

import os
import time
import uuid
import json
import base64
import shutil
from typing import Dict, Any

import streamlit as st

from config import UPLOAD_FOLDER, VECTOR_STORE_PATH, STORAGE_PATH, MAX_QUESTIONS_PER_SESSION
from utils.pdf_loader import load_pdf
from utils.page_index_chunker import chunk_by_page_index
from utils.layout_extractor import extract_layout
from utils.hierarchical_chunker import build_hierarchy
from utils.tenant_vector_store import TenantVectorStore
from utils.vector_store import create_vector_store
from utils.rag import stream_answer
from utils.ragas_scorer import score_response

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AM RAG Document QA",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Registry helpers ───────────────────────────────────────────────────────────
REGISTRY_FILE = os.path.join(VECTOR_STORE_PATH, "registry.json")


def load_registry() -> Dict[str, Any]:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_registry(registry: Dict[str, Any]) -> None:
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


# ── Session state initialisation ───────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]

if "documents" not in st.session_state:
    st.session_state.documents = load_registry()

if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0


# ── PDF processing ─────────────────────────────────────────────────────────────

def process_pdf(uploaded_file, strategy: str) -> str:
    """
    Save the uploaded PDF, extract, chunk, embed, and build the appropriate index.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        strategy:      "page_index" or "parent_child".

    Returns:
        doc_id assigned to this document.
    """
    doc_id = uuid.uuid4().hex[:8]
    safe_name = uploaded_file.name.replace(" ", "_")
    file_path = os.path.join(UPLOAD_FOLDER, f"{doc_id}_{safe_name}")

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    if strategy == "parent_child":
        # ── Parent-Child pipeline ──────────────────────────────────────────────
        pages = extract_layout(file_path)          # layout-aware Markdown extraction
        parents, children = build_hierarchy(pages, doc_id, uploaded_file.name)

        session_id = st.session_state.session_id
        store = TenantVectorStore(session_id)
        store.build(parents, children, doc_id)

        st.session_state.documents[doc_id] = {
            "name": uploaded_file.name,
            "path": file_path,
            "pages": len(pages),
            "chunks": len(children),
            "parents": len(parents),
            "chunking_strategy": "parent_child",
            "session_id": session_id,
        }
    else:
        # ── PageIndex RAG pipeline ─────────────────────────────────────────────
        pages = load_pdf(file_path)
        chunks = chunk_by_page_index(file_path, doc_id)

        actual_strategy = (
            chunks[0].metadata.get("chunking_strategy", "fixed") if chunks else "fixed"
        )
        sections = len({
            c.metadata.get("section_title") for c in chunks
            if c.metadata.get("section_title")
        })

        create_vector_store(chunks, doc_id)

        st.session_state.documents[doc_id] = {
            "name": uploaded_file.name,
            "path": file_path,
            "pages": len(pages),
            "chunks": len(chunks),
            "chunking_strategy": actual_strategy,
            "sections": sections,
        }

    save_registry(st.session_state.documents)
    return doc_id


# ── Metrics panel helper ───────────────────────────────────────────────────────

def _score_emoji(v: float | None) -> str:
    if v is None:
        return ""
    return "🟢" if v >= 0.7 else ("🟡" if v >= 0.5 else "🔴")


def _render_metrics_panel(
    meta: dict, total_ms, sources: list, ls_data: dict, ragas_scores: dict = {}
) -> None:
    """Render the 📊 Query Metrics expander for one chat turn."""
    with st.expander("📊 Query Metrics", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("⏱️ Total", f"{total_ms} ms" if total_ms else "—")
        c2.metric("⚙️ Retrieval", f"{meta.get('retrieval_ms', '—')} ms")
        c3.metric("📦 Retrieved", len(sources))

        if meta.get("retry_count", 0) > 0:
            rewritten = meta.get("rewritten_question", "")
            st.info(
                f"🔄 Query rewritten {meta['retry_count']}×"
                + (f" → *\"{rewritten}\"*" if rewritten else "")
            )

        if ls_data.get("total_tokens"):
            st.caption(
                f"🔢 Tokens: **{ls_data['total_tokens']}** "
                f"(↑ {ls_data.get('prompt_tokens', '?')} in · "
                f"↓ {ls_data.get('completion_tokens', '?')} out)"
            )

        if ragas_scores:
            st.divider()
            if "_error" in ragas_scores:
                st.warning(f"🔬 RAGAS scoring failed: {ragas_scores['_error']}")
            else:
                faith = ragas_scores.get("faithfulness")
                relev = ragas_scores.get("answer_relevancy")
                r1, r2 = st.columns(2)
                r1.metric(
                    f"{_score_emoji(faith)} Faithfulness",
                    f"{faith:.2f}" if faith is not None else "—",
                    help="Does the answer use only information from the retrieved chunks?",
                )
                r2.metric(
                    f"{_score_emoji(relev)} Answer Relevancy",
                    f"{relev:.2f}" if relev is not None else "—",
                    help="Is the answer directly relevant to the question?",
                )


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    col_logo, col_title = st.columns([1, 3])
    with col_logo:
        with open("logo.jpeg", "rb") as _f:
            _b64 = base64.b64encode(_f.read()).decode()
        st.markdown(
            f'<img src="data:image/jpeg;base64,{_b64}" width="75" '
            f'style="border-radius:50%;pointer-events:none;user-select:none;">',
            unsafe_allow_html=True,
        )
    with col_title:
        st.markdown("**📄 AM RAG Document QA**")
        st.caption("Ask questions about your PDF documents using AI.")
    st.divider()

    # ── Live RAGAS toggle ──────────────────────────────────────────────────────
    ragas_live = st.toggle(
        "🔬 Live RAGAS scoring",
        value=st.session_state.get("ragas_live", False),
        help="Score each answer with Faithfulness + Answer Relevancy (~10 s extra per query)",
    )
    st.session_state.ragas_live = ragas_live
    if ragas_live:
        st.caption("🟢 ≥ 0.7 good · 🟡 0.5–0.69 borderline · 🔴 < 0.5 poor")

    # ── Session question quota ──────────────────────────────────────────────────
    _used = st.session_state.question_count
    st.caption(f"💬 Questions this session: **{_used} / {MAX_QUESTIONS_PER_SESSION}**")
    st.progress(min(_used / MAX_QUESTIONS_PER_SESSION, 1.0))

    st.divider()

    # ── Upload widget ──────────────────────────────────────────────────────────
    st.subheader("1 · Upload a PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Works best with text-based PDFs (not scanned images).",
        label_visibility="collapsed",
    )

    # ── Chunking strategy selector ─────────────────────────────────────────────
    chosen_strategy = st.radio(
        "Chunking Strategy",
        options=["page_index", "parent_child"],
        format_func=lambda s: (
            "🗂️ PageIndex RAG — section-aware heading detection"
            if s == "page_index"
            else "👨‍👩‍👦 Parent-Child — hierarchical, layout-aware, multi-tenant"
        ),
        index=0,
        help=(
            "PageIndex: detects headings and chunks by section.\n"
            "Parent-Child: splits into large parent blocks + small child embeddings "
            "for precise retrieval with full-context LLM answers."
        ),
    )

    if uploaded_file is not None:
        if st.button("⚙️ Process PDF", type="primary", use_container_width=True):
            spinner_label = (
                "Extracting layout, building parent-child index…"
                if chosen_strategy == "parent_child"
                else "Extracting text, building index…"
            )
            with st.spinner(spinner_label):
                try:
                    doc_id = process_pdf(uploaded_file, chosen_strategy)
                    info = st.session_state.documents[doc_id]

                    st.success("✅ Done!")

                    if chosen_strategy == "parent_child":
                        st.info(
                            f"📑 Pages: {info['pages']}  \n"
                            f"👨‍👩‍👦 Parents: {info['parents']}  \n"
                            f"🧩 Children: {info['chunks']}  \n"
                            f"⚙️ Strategy: Parent-Child RAG"
                        )
                    else:
                        strategy_label = (
                            "PageIndex RAG"
                            if info.get("chunking_strategy") == "page_index"
                            else "Fixed-size"
                        )
                        sections_line = (
                            f"  \n📂 Sections: {info['sections']}"
                            if info.get("chunking_strategy") == "page_index" else ""
                        )
                        st.info(
                            f"📑 Pages: {info['pages']}  \n"
                            f"🧩 Chunks: {info['chunks']}  \n"
                            f"⚙️ Strategy: {strategy_label}"
                            f"{sections_line}"
                        )

                    st.session_state.selected_doc_id = doc_id
                    st.session_state.chat_history = []
                    st.rerun()
                except ValueError as exc:
                    st.error(f"❌ {exc}")
                except Exception as exc:
                    st.error(f"❌ Unexpected error: {exc}")

    st.divider()

    # ── Document list ──────────────────────────────────────────────────────────
    st.subheader("2 · Select a document")

    if not st.session_state.documents:
        st.info("No documents yet. Upload one above.")
    else:
        for doc_id, info in list(st.session_state.documents.items()):
            is_active = st.session_state.selected_doc_id == doc_id
            label = f"{'✅ ' if is_active else '📄 '}{info['name']}"
            col_select, col_delete = st.columns([5, 1])
            with col_select:
                if st.button(label, key=f"select_{doc_id}", use_container_width=True):
                    if st.session_state.selected_doc_id != doc_id:
                        st.session_state.selected_doc_id = doc_id
                        st.session_state.chat_history = []
                        st.rerun()
            with col_delete:
                if st.button("🗑️", key=f"delete_{doc_id}", help=f"Remove {info['name']}"):
                    doc_strategy = info.get("chunking_strategy", "fixed")

                    if doc_strategy == "parent_child":
                        # Remove from TenantVectorStore (FAISS + parent JSON)
                        doc_session = info.get("session_id", st.session_state.session_id)
                        tvs = TenantVectorStore(doc_session)
                        tvs.delete_doc(doc_id)
                    else:
                        # Remove PageIndex FAISS folder
                        store_path = os.path.join(VECTOR_STORE_PATH, doc_id)
                        if os.path.exists(store_path):
                            shutil.rmtree(store_path)

                    if os.path.exists(info.get("path", "")):
                        os.remove(info["path"])

                    if st.session_state.selected_doc_id == doc_id:
                        st.session_state.selected_doc_id = None
                        st.session_state.chat_history = []

                    del st.session_state.documents[doc_id]
                    save_registry(st.session_state.documents)
                    st.rerun()

    st.divider()
    st.caption(
        "**Embeddings:** all-MiniLM-L6-v2  \n"
        "**LLM:** DeepSeek v4-flash  \n"
        "**Stack:** LangChain · LangGraph · LangSmith"
    )



# ── Main area ──────────────────────────────────────────────────────────────────

st.title("🤖 AM RAG Document Question & Answer")

if st.session_state.selected_doc_id is None:
    st.info("👈 Upload a PDF from the sidebar and select it to start chatting.")

    with st.expander("📖 How it works", expanded=True):
        tab_pipeline, tab_strategies, tab_stack = st.tabs(
            ["🔄 Pipeline", "🗂️ Chunking Strategies", "🛠️ Tech Stack"]
        )

        with tab_pipeline:
            st.markdown("#### Corrective Agentic RAG")
            st.caption(
                "A self-correcting LangGraph pipeline — if retrieved context is irrelevant, "
                "the query is automatically rewritten and retrieval retried (max 2×). "
                "Every node is traced in LangSmith automatically."
            )
            st.markdown(
                """
<div style="font-family:sans-serif;line-height:1.6;">

  <div style="background:#e8f4fd;border-left:4px solid #1f77b4;
              border-radius:6px;padding:12px 16px;margin:6px 0;">
    <b>① Retrieve</b><br>
    <span style="font-size:0.9em;color:#444;">
      Embeds the question and searches FAISS for the most relevant chunks.<br>
      <i>PageIndex</i> returns 7 section chunks &nbsp;·&nbsp;
      <i>Parent-Child</i> retrieves 3 large parent blocks via child-vector lookup.
    </span>
  </div>

  <div style="text-align:center;color:#aaa;font-size:1.3em;margin:2px 0;">↓</div>

  <div style="background:#fff8e1;border-left:4px solid #f5a623;
              border-radius:6px;padding:12px 16px;margin:6px 0;">
    <b>② Grade Chunks</b><br>
    <span style="font-size:0.9em;color:#444;">
      LLM judges whether the retrieved context actually answers the question.
    </span>
  </div>

  <div style="text-align:center;color:#aaa;font-size:1.3em;margin:2px 0;">↓</div>

  <div style="display:flex;gap:10px;margin:6px 0;">
    <div style="flex:1;background:#e9f7ef;border-left:4px solid #28a745;
                border-radius:6px;padding:12px 16px;">
      <b>✅ Relevant → Generate</b><br>
      <span style="font-size:0.9em;color:#444;">
        LLM streams a grounded answer strictly from the retrieved context.
      </span>
    </div>
    <div style="flex:1;background:#fdecea;border-left:4px solid #e74c3c;
                border-radius:6px;padding:12px 16px;">
      <b>🔄 Irrelevant → Rewrite</b><br>
      <span style="font-size:0.9em;color:#444;">
        LLM reformulates the query → retrieve again (up to 2 retries).
      </span>
    </div>
  </div>

</div>
""",
                unsafe_allow_html=True,
            )

        with tab_strategies:
            col_a, col_b = st.columns(2, gap="medium")
            with col_a:
                st.markdown("**🗂️ PageIndex RAG** *(default)*")
                st.markdown(
                    "- Detects bold/font headings via PyMuPDF\n"
                    "- Chunks by section with `[Section: title]` prefix\n"
                    "- 1 500 char chunks · **7 retrieved** per query\n"
                    "- Falls back to fixed-size if < 3 sections found"
                )
            with col_b:
                st.markdown("**👨‍👩‍👦 Parent-Child RAG**")
                st.markdown(
                    "- Layout-aware extraction — tables preserved as Markdown\n"
                    "- Child chunks (1 000 chars) embedded for precision\n"
                    "- Parent blocks (6 000 chars) returned to LLM for full context\n"
                    "- Session-isolated FAISS · **3 parent blocks** per query"
                )

        with tab_stack:
            st.markdown(
                "| Component | Technology |\n"
                "|:---|:---|\n"
                "| Graph orchestration | **LangGraph** StateGraph |\n"
                "| Chains | **LangChain** LCEL (`prompt | llm | parser`) |\n"
                "| Chunking A | **PageIndex RAG** — section-aware heading detection |\n"
                "| Chunking B | **Parent-Child** — hierarchical, layout-aware |\n"
                "| Embeddings | `all-MiniLM-L6-v2` (local, CPU) |\n"
                "| Vector store | **FAISS** (isolated per session + document) |\n"
                "| LLM | **DeepSeek v4-flash** |\n"
                "| Observability | **LangSmith** (backend tracing) |\n"
            )
else:
    active_id = st.session_state.selected_doc_id
    active_info = st.session_state.documents.get(active_id, {})
    doc_strategy = active_info.get("chunking_strategy", "fixed")

    # ── Active document banner ─────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
    with col1:
        st.markdown(f"**Active document:** `{active_info.get('name', active_id)}`")
    with col2:
        st.caption(f"📑 {active_info.get('pages', '?')} pages")
    with col3:
        if doc_strategy == "parent_child":
            st.caption(f"🧩 {active_info.get('chunks', '?')} children")
        else:
            st.caption(f"🧩 {active_info.get('chunks', '?')} chunks")
    with col4:
        if doc_strategy == "parent_child":
            st.caption("👨‍👩‍👦 Parent-Child")
        elif doc_strategy == "page_index":
            st.caption("🗂️ PageIndex")
        else:
            st.caption("⚙️ Fixed-size")

    st.divider()

    # ── Replay chat history ────────────────────────────────────────────────────
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])

        with st.chat_message("assistant"):
            st.write(turn["answer"])
            _render_sources = turn.get("sources", [])
            if _render_sources:
                label = (
                    f"📚 Parent blocks ({len(_render_sources)})"
                    if doc_strategy == "parent_child"
                    else f"📚 Sources ({len(_render_sources)} chunks)"
                )
                with st.expander(label):
                    for i, src in enumerate(_render_sources, 1):
                        section = src.get("section_title")
                        child_count = src.get("child_count")
                        hdr = f"**{'Block' if doc_strategy == 'parent_child' else 'Chunk'} {i} — Page {src['page']}**"
                        if section:
                            hdr += f" · *{section}*"
                        if child_count:
                            hdr += f" · {child_count} children"
                        st.markdown(hdr)
                        preview = src["text"]
                        if len(preview) > 800:
                            preview = preview[:800] + "…"
                        st.text(preview)
                        st.divider()

            # Metrics panel (replayed from stored history)
            _meta = turn.get("meta", {})
            _total_ms = turn.get("total_ms")
            if _meta or _total_ms:
                _render_metrics_panel(
                    meta=_meta,
                    total_ms=_total_ms,
                    sources=_render_sources,
                    ls_data=turn.get("ls_data", {}),
                    ragas_scores=turn.get("ragas_scores", {}),
                )

    # ── Question input ─────────────────────────────────────────────────────────
    _limit_reached = st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION
    if _limit_reached:
        st.info(
            f"🚦 You've reached the demo limit of {MAX_QUESTIONS_PER_SESSION} questions "
            "for this session. Refresh the page to start a new session."
        )
    question = st.chat_input(
        "Ask a question about this document…"
        if not _limit_reached
        else "Session question limit reached — refresh to reset",
        disabled=_limit_reached,
    )

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                spinner_label = (
                    "Retrieving child vectors → looking up parent blocks…"
                    if doc_strategy == "parent_child"
                    else "Retrieving & grading chunks…"
                )
                t_start = time.time()
                with st.spinner(spinner_label):
                    token_stream, sources, meta = stream_answer(
                        doc_id=active_id,
                        question=question,
                        session_id=st.session_state.session_id,
                        doc_strategy=doc_strategy,
                    )

                answer = st.write_stream(token_stream)
                total_ms = int((time.time() - t_start) * 1000)

                if sources:
                    label = (
                        f"📚 Parent blocks ({len(sources)})"
                        if doc_strategy == "parent_child"
                        else f"📚 Sources ({len(sources)} chunks)"
                    )
                    with st.expander(label):
                        for i, src in enumerate(sources, 1):
                            section = src.get("section_title")
                            child_count = src.get("child_count")
                            hdr = f"**{'Block' if doc_strategy == 'parent_child' else 'Chunk'} {i} — Page {src['page']}**"
                            if section:
                                hdr += f" · *{section}*"
                            if child_count:
                                hdr += f" · {child_count} children"
                            st.markdown(hdr)
                            preview = src["text"]
                            if len(preview) > 800:
                                preview = preview[:800] + "…"
                            st.text(preview)
                            st.divider()

                # ── Optional LangSmith run fetch ───────────────────────────────
                ls_data = {}
                if os.getenv("LANGCHAIN_API_KEY"):
                    try:
                        from langsmith import Client
                        _ls_client = Client()
                        _runs = list(_ls_client.list_runs(
                            project_name=os.getenv("LANGCHAIN_PROJECT", "simple-doc-rag"),
                            run_type="chain",
                            limit=1,
                        ))
                        if _runs:
                            _r = _runs[0]
                            ls_data = {
                                "total_tokens":      getattr(_r, "total_tokens", None),
                                "prompt_tokens":     getattr(_r, "prompt_tokens", None),
                                "completion_tokens": getattr(_r, "completion_tokens", None),
                                "url":               getattr(_r, "url", None),
                            }
                    except Exception:
                        pass   # best-effort — never break the UI

                # ── Optional live RAGAS scoring ────────────────────────────────
                ragas_scores = {}
                if st.session_state.get("ragas_live") and answer and sources:
                    contexts = [s["text"] for s in sources if s.get("text")]
                    with st.spinner("🔬 Scoring with RAGAS (faithfulness + relevancy)…"):
                        ragas_scores = score_response(question, answer, contexts)

                # ── Query metrics panel ────────────────────────────────────────
                _render_metrics_panel(meta, total_ms, sources, ls_data, ragas_scores)

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                    "meta": meta,
                    "total_ms": total_ms,
                    "ls_data": ls_data,
                    "ragas_scores": ragas_scores,
                })

                # Count only successful answers; rerun so the sidebar counter
                # and the input gate reflect the new total immediately.
                st.session_state.question_count += 1
                st.rerun()

            except FileNotFoundError as exc:
                st.error(f"❌ {exc}")
            except Exception as exc:
                st.error(f"❌ Error: {exc}")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
