---
title: AM RAG Document QA
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---

# AM RAG Document QA

*Prepared by Ammar Nasir | AI Engineer*

A production-quality **Corrective Agentic RAG** system built to showcase modern Generative AI engineering skills — LangChain, LangGraph, LangSmith, RAG, Agentic AI, and **RAGAS evaluation**.

Upload any PDF → ask questions → get answers grounded strictly in that document, with automatic query rewriting when retrieval quality is poor, two selectable chunking strategies, live answer-quality scoring, and per-query metrics.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI** | Streamlit |
| **Graph Orchestration** | LangGraph `StateGraph` |
| **Chains** | LangChain LCEL (`PromptTemplate \| LLM \| StrOutputParser`) |
| **Chunking Strategy A** | **PageIndex RAG** — section-aware heading detection |
| **Chunking Strategy B** | **Parent-Child RAG** — hierarchical, layout-aware, multi-tenant |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, CPU) |
| **Vector Store** | FAISS (PageIndex: per-document · Parent-Child: session-isolated) |
| **PDF Parsing** | PyMuPDF + `pymupdf4llm` (layout/table-aware Markdown) |
| **Evaluation** | **RAGAS** (Faithfulness, Answer Relevancy, Context Precision/Recall) |
| **LLM (primary)** | DeepSeek v4-flash (OpenAI-compatible API) |
| **LLM (alternative)** | Ollama (local) |
| **LLM (fallback)** | HuggingFace `flan-t5-base` (no API key needed) |
| **Observability** | LangSmith (auto-traced via LCEL + LangGraph) |

---

## RAG Type: Corrective Agentic RAG

This is not a simple Naive RAG pipeline. It implements **Corrective RAG (CRAG)** orchestrated by a **LangGraph StateGraph**, making the retrieval process agentic and self-correcting:

```
[Question]
     ↓
 retrieve        ←  strategy dispatch (PageIndex FAISS  OR  child→parent lookup)
     ↓
grade_chunks     ←  LLM judges if chunks are relevant
     ↓
 relevant?
 /        \
yes        no (retry ≤ 2×)
 ↓              ↓
generate    rewrite_query  ←  LLM rewrites question
                 ↓
              retrieve     ←  searches again
                 ↓
              generate     ←  LCEL chain → streamed answer
```

The `retrieve` node **dispatches per strategy**: PageIndex performs section-aware FAISS search; Parent-Child embeds and searches small child chunks, then returns their larger parent blocks for full context.

**If retrieved chunks are irrelevant**, the LLM automatically rewrites the question and retries retrieval before generating — no fixed pipeline, the LLM drives the flow.

---

## Chunking Strategies

The strategy is chosen per document at upload time.

| | 🗂️ **PageIndex RAG** *(default)* | 👨‍👩‍👦 **Parent-Child RAG** |
|---|---|---|
| **Idea** | Detect bold/font headings, chunk by section | Embed small child chunks, return large parent blocks |
| **Extraction** | PyMuPDF text + heading detection | `pymupdf4llm` layout-aware (tables → Markdown) |
| **Chunk size** | 1 500 chars (`[Section: title]` prefixed) | child 1 000 / parent 6 000 chars |
| **Retrieved** | `PAGE_INDEX_TOP_K = 7` sections | `PARENT_CHILD_TOP_K = 3` parent blocks |
| **Vector store** | Per-document FAISS | Session-isolated FAISS (`TenantVectorStore`) |
| **Best for** | Well-structured docs with clear headings | Dense docs, tables, where precise match + full context both matter |
| **Fallback** | Fixed-size chunking if < 3 sections found | — |

**Why Parent-Child?** Small child chunks give precise semantic matches; returning the larger parent block gives the LLM full surrounding context — solving the "context split" problem of plain fixed-size chunking.

---

## Evaluation (RAGAS)

Answer quality is measured with [RAGAS](https://docs.ragas.io), using DeepSeek as the LLM judge and the same local MiniLM embeddings as the app.

### Live in-app scoring

Toggle **🔬 Live RAGAS scoring** in the sidebar. After each answer, the **📊 Query Metrics** panel shows:

- **Faithfulness** — does the answer use only information from the retrieved chunks?
- **Answer Relevancy** — is the answer directly relevant to the question?

Scored with a colour legend: 🟢 ≥ 0.7 good · 🟡 0.5–0.69 borderline · 🔴 < 0.5 poor.
*(Adds ~10 s per query — off by default.)*

### Batch evaluation CLI — `evaluate.py`

Runs the **real LangGraph pipeline** (not a reimplementation) against a set of questions:

```bash
# List indexed documents
python evaluate.py --list

# Faithfulness + Answer Relevancy (no reference answers needed)
python evaluate.py --doc "vector vs vectorless RAG" \
    --questions "What is Vector RAG?" "What are its limitations?"

# Add Context Precision + Context Recall (needs ground-truth answers)
python evaluate.py --doc "vector vs vectorless RAG" \
    --questions "What is Vector RAG?" \
    --ground_truths "Vector RAG uses embeddings and FAISS to retrieve chunks..."
```

| Metric | Measures | Needs ground truth? |
|--------|----------|---------------------|
| `faithfulness` | Answer uses only retrieved context | No |
| `answer_relevancy` | Answer is on-topic to the question | No |
| `context_precision` | Retrieved chunks ranked by relevance | Yes |
| `context_recall` | Context covers the reference answer | Yes |

Results print as a table and save to `results/ragas_eval.csv`.

---

## Features

- **Two chunking strategies** — PageIndex (section-aware) and Parent-Child (hierarchical, layout-aware)
- Upload multiple PDFs — each gets its own isolated index
- **Delete documents** directly from the sidebar (removes index + PDF)
- **Layout/table-aware extraction** via `pymupdf4llm` (Parent-Child strategy)
- **Multi-tenant isolation** — Parent-Child vectors are scoped per browser session
- **Agentic retrieval grading** — LLM checks chunk relevance before answering
- **Automatic query rewriting** — poor retrieval triggers a smarter retry (≤ 2×)
- **Streaming responses** — answer appears token-by-token
- **📊 Per-query metrics** — total & retrieval latency, chunks retrieved, token usage, rewrite count
- **🔬 Live RAGAS scoring** — optional per-answer Faithfulness + Answer Relevancy
- **Per-session question quota** — configurable demo rate limit (default 12)
- Source chunks with page numbers shown below every answer
- Chat history within a session
- Document registry persists across restarts
- **Full LangSmith observability** — every graph node auto-traced

---

## Project Structure

```
RAG/
├── app.py                      # Streamlit UI (upload, chat, metrics, RAGAS toggle, quota)
├── config.py                   # All configuration — edit here, nowhere else
├── evaluate.py                 # RAGAS batch evaluation CLI (runs the real pipeline)
├── requirements.txt
├── README.md
├── logo.jpeg                   # App logo (sidebar)
├── .env                        # API keys (not committed)
├── .env.example                # Key reference template
├── uploads/                    # Saved PDFs (auto-created)
├── vector_store/               # PageIndex FAISS indices + registry.json (auto-created)
├── storage/                    # Parent-Child parents + session-isolated vectors (auto-created)
├── results/                    # RAGAS evaluation CSVs (auto-created)
└── utils/
    ├── __init__.py
    ├── pdf_loader.py            # PyMuPDF text extraction (page-aware)
    ├── embeddings.py           # HuggingFace singleton embedding model
    ├── llm.py                  # Multi-provider LLM factory (DeepSeek / Ollama / HF)
    ├── vector_store.py         # PageIndex FAISS create / load / search / retriever
    ├── page_index_chunker.py   # Section-aware heading-based chunking (PageIndex)
    ├── layout_extractor.py     # Layout/table-aware Markdown extraction (Parent-Child)
    ├── hierarchical_chunker.py # Builds parent + child chunks
    ├── parent_store.py         # JSON persistence for parent blocks
    ├── tenant_vector_store.py  # Session-isolated FAISS for Parent-Child
    ├── graph.py                # LangGraph StateGraph — agentic RAG pipeline
    ├── rag.py                  # Public API: answer_question() + stream_answer()
    └── ragas_scorer.py         # Live RAGAS scoring (Faithfulness + Answer Relevancy)
```

---

## Prerequisites

### Python 3.10+

```bash
python --version   # 3.10 or newer required (developed on 3.11)
```

### DeepSeek API Key (primary LLM)

1. Sign up at **https://platform.deepseek.com**
2. Create an API key
3. Add it to `.env` (see [Configuration](#configuration))

> **Tip for public demos:** set a hard monthly spend cap on your DeepSeek key. The in-app session quota is a courtesy limit; the spend cap is the real cost ceiling.

> **No DeepSeek?** Set `LLM_PROVIDER = "ollama"` in `config.py` and install Ollama (`ollama pull llama3.2`), or use `"huggingface"` for a fully local fallback.

### LangSmith API Key (optional — observability)

1. Sign up free at **https://smith.langchain.com** (5 000 traces/month free)
2. Create an API key and add to `.env`

---

## Installation

```bash
# 1. Clone / download the project
cd RAG

# 2. Create a virtual environment (Python 3.10+)
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies (includes ragas + datasets for evaluation)
pip install -r requirements.txt
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your keys:

```env
# LLM
DEEPSEEK_API_KEY=your-deepseek-api-key-here

# LangSmith (optional — enables full observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key-here
LANGCHAIN_PROJECT=simple-doc-rag
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key-here
LANGSMITH_PROJECT=simple-doc-rag
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

All tunable parameters live in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `"deepseek"` | `"deepseek"` / `"ollama"` / `"huggingface"` |
| `DEEPSEEK_MODEL` | `"deepseek-v4-flash"` | DeepSeek model name |
| `OLLAMA_MODEL` | `"llama3.2"` | Any model pulled via Ollama |
| `CHUNK_SIZE` | `1500` | Characters per chunk (PageIndex / fixed) |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `PARENT_CHUNK_SIZE` | `6000` | Parent block size (Parent-Child) |
| `CHILD_CHUNK_SIZE` | `1000` | Child chunk size (Parent-Child) |
| `CHILD_CHUNK_OVERLAP` | `200` | Child chunk overlap |
| `TOP_K` | `5` | Default fallback retrieval count |
| `PAGE_INDEX_TOP_K` | `7` | Sections retrieved (PageIndex) |
| `PARENT_CHILD_TOP_K` | `3` | Parent blocks retrieved (Parent-Child) |
| `MAX_QUESTIONS_PER_SESSION` | `12` | Per-session demo question quota |
| `EMBEDDING_MODEL` | `"sentence-transformers/all-MiniLM-L6-v2"` | HuggingFace embedding model |

---

## Running the App

```bash
streamlit run app.py
```

App opens at **http://localhost:8501**.

---

## Example Workflow

```
1.  Launch:      streamlit run app.py
2.  Sidebar  →   (optional) toggle 🔬 Live RAGAS scoring
3.  Sidebar  →   Upload a PDF → pick a chunking strategy → click "Process PDF"
4.  Sidebar  →   Click the document name to select it
5.  Main     →   Type a question in the chat box
6.  Graph:       retrieve → grade_chunks → (generate OR rewrite → retrieve → generate)
7.  Response:    Answer streams token-by-token + 📚 sources + 📊 query metrics (+ RAGAS scores)
8.  Sidebar  →   Watch the question quota counter; click 🗑️ to remove a document
```

---

## LangSmith Observability

With `LANGSMITH_API_KEY` set, every query is automatically traced at **https://smith.langchain.com**:

- **`retrieve`** node — search inputs/outputs + latency
- **`grade_chunks`** node — LLM relevance decision + token usage
- **`rewrite_query`** node — rewritten question (fires only when chunks are poor)
- **`generate`** node — full prompt, streamed answer, token count, cost

No manual instrumentation needed — LangGraph + LCEL trace automatically. The trace URL is intentionally **not** exposed in the UI (developer-only).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `DEEPSEEK_API_KEY` error | Add key to `.env` or switch `LLM_PROVIDER` to `"ollama"` |
| `Connection refused` on Ollama | Run `ollama serve` |
| `model not found` | Run `ollama pull llama3.2` |
| `No text extracted` | PDF may be scanned; try an OCR version |
| Slow first question | Embedding model loads once on first query (~90 MB download) |
| `RAGAS scoring failed: No module named 'ragas'` | Install into the **running** env: `pip install -r requirements.txt` |
| Code change not taking effect | `.streamlit/config.toml` disables auto-reload — fully restart Streamlit |
| Answer says "I could not find..." | Re-upload and reprocess the PDF — index may be missing |
| `faiss` import error | Run `pip install faiss-cpu` |

---

## Skills Demonstrated

| Skill | Implementation |
|-------|---------------|
| **LangChain** | LCEL chains, PromptTemplate, HuggingFaceEmbeddings, FAISS vectorstore, ChatOpenAI |
| **LangGraph** | `StateGraph`, typed state (`RAGState`), conditional edges, agentic retry loop |
| **LangSmith** | Auto-tracing via LCEL callbacks, project-level observability, token/cost tracking |
| **RAG** | Corrective RAG (CRAG) — retrieval grading + query rewriting |
| **Advanced retrieval** | Two strategies — section-aware PageIndex and hierarchical Parent-Child |
| **RAGAS evaluation** | Live in-app scoring + batch CLI (faithfulness, relevancy, context precision/recall) |
| **Multi-tenant design** | Session-isolated vector stores and parent stores |
| **Agentic AI** | LLM-driven flow control — the model decides whether to retrieve again |
| **Productionisation** | Per-session rate limiting, query metrics, layout/table-aware parsing |

---

## Limitations

- No authentication or multi-user accounts (vectors are isolated per session, not per user)
- Chat history and the question quota are session-only (reset on browser refresh)
- Live RAGAS scoring adds ~10 s per query (off by default)
- Scanned/image-only PDFs are not supported (no OCR)
- HuggingFace fallback (`flan-t5-base`) produces shorter answers than DeepSeek

---

**Prepared by Ammar Nasir | AI Engineer**
