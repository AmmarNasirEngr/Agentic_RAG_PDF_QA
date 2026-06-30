# App Running Guide

A quick guide to get **AM RAG Document QA** running locally.

---

## 1. Prerequisites

- **Python 3.10+** (developed on 3.11)
- A **DeepSeek API key** → https://platform.deepseek.com
- *(Optional)* a **LangSmith API key** for tracing → https://smith.langchain.com

---

## 2. Setup (one time)

```bash
cd RAG

# Create & activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (PowerShell)
# source .venv/bin/activate         # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 3. Add your API key

Copy `.env.example` → `.env` and fill in:

```env
DEEPSEEK_API_KEY=your-deepseek-api-key-here
```

*(LangSmith keys are optional — leave blank to skip tracing.)*

---

## 4. Run

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**.

---

## 5. Use it

1. *(Optional)* toggle **🔬 Live RAGAS scoring** in the sidebar.
2. **Upload a PDF** → pick a chunking strategy → click **Process PDF**.
3. Click the document name to **select** it.
4. Ask a question in the chat box.
5. Read the streamed answer + **📚 sources** + **📊 query metrics**.

---

## 6. Evaluate (optional, CLI)

```bash
python evaluate.py --list

python evaluate.py --doc "your document name" \
    --questions "Question 1?" "Question 2?"
```

Results save to `results/ragas_eval.csv`.

---

## Common Issues

| Problem | Fix |
|---------|-----|
| Code change not showing | Auto-reload is off — **fully restart** Streamlit (`Ctrl+C`, then rerun) |
| `No module named 'ragas'` | Run `pip install -r requirements.txt` in the **active** `.venv` |
| `DEEPSEEK_API_KEY` error | Check the key is set in `.env` |
| Slow first question | Embedding model downloads once (~90 MB) on first query |
| `faiss` import error | `pip install faiss-cpu` |

---

*Prepared by Ammar Nasir | AI Engineer*
