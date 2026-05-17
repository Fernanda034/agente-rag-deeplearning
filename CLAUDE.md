# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Copy `.env.example` to `.env` and add a valid Groq API key:

```bash
cp .env.example .env
# Edit .env: GROQ_API_KEY=<your_key>
```

Install dependencies (Python 3.10 or 3.11):

```bash
pip install -r requirements_chroma_py39.txt   # minimal: chromadb + pypdf + sentence-transformers
pip install -r requirements.txt               # full: LangChain + Streamlit + Groq + RAGAS
```

## Key Commands

**Build the vector database (must run before the agent or app):**

```bash
python 01_ingesta/ingestion_chroma_direct.py --embedding-backend onnx-minilm
```

**Preview chunking without writing to ChromaDB:**

```bash
python 01_ingesta/preprocess_preview.py
```

**Search the vector DB by topic:**

```bash
python 02_vectorstore/retriever_chroma_direct.py "Que es Batch Normalization?" --topic regularizacion_estabilidad
python 02_vectorstore/retriever_chroma_direct.py "Por que ResNet usa conexiones residuales?" --topic cnn_vision
```

**Run the Streamlit chat app:**

```bash
streamlit run app.py
```

**Evaluate corpus coverage:**

```bash
python 04_evaluacion/evaluar_cobertura_direct.py --embedding-backend onnx-minilm -k 5
python 04_evaluacion/evaluar_profundidad_chunks.py --embedding-backend onnx-minilm -k 5
```

## Architecture

### Two ingestion paths

There are two parallel ingestion implementations that should not be confused:

| File | Backend | Use |
|---|---|---|
| `01_ingesta/ingestion.py` | LangChain + HuggingFaceEmbeddings | Original; uses `SQLRecordManager` for incremental dedup |
| `01_ingesta/ingestion_chroma_direct.py` | ChromaDB directly + ONNX MiniLM | Current preferred path; no LangChain dependency for ingestion |

Similarly, `02_vectorstore/retriever_chroma_direct.py` is the direct ChromaDB retriever, while `02_vectorstore/retriever_metadata.py` is the LangChain-based counterpart.

### RAG data flow

```
corpus/ PDFs
  → docs/metadatos_temas_corpus.csv  (manifest: branch/topic/subtopic per source)
  → 01_ingesta/ingestion_chroma_direct.py  (PDF text extraction, cleaning, chunking)
  → ChromaDB at vector_db/  (chunks with metadata + ONNX MiniLM embeddings)
  → rag_utils/query_expansion.py  (bilingual ES→EN term expansion)
  → 02_vectorstore/retriever_chroma_direct.py  (filtered retrieval by branch/topic/subtopic)
  → 03_agente/agent.py  (LangChain ReAct agent: retriever tool + ArXiv fallback)
  → app.py  (Streamlit chat UI)
```

### Metadata schema

Each chunk stored in ChromaDB carries: `branch`, `topic`, `subtopic`, `chapter`, `source`, `file_name`, `page`, `source_type`, `scope`, `use_in_retrieval`, `priority`. These fields come from `docs/metadatos_temas_corpus.csv`. Sources with `use_in_retrieval=no` are skipped at ingestion time.

The two active branches are `deep_learning` (primary) and `machine_learning_estadistica` (secondary/support).

### Embedding consistency

`rag_utils/onnx_embeddings.py` provides `ChromaONNXEmbeddings`, a LangChain-compatible wrapper around ChromaDB's bundled ONNX MiniLM-L6-v2. Both `03_agente/agent.py` and `02_vectorstore/test_retrieval.py` use this class so that query vectors are produced by the same model implementation as the stored document vectors (ingested via `ingestion_chroma_direct.py`). **Do not swap this for `HuggingFaceEmbeddings` in the agent** — ONNX and PyTorch implementations of the same model can produce numerically different vectors, silently degrading retrieval.

### Agent tools and retrieval pipeline

`03_agente/agent.py` builds a LangChain tool-calling agent (Groq `llama-3.3-70b-versatile`) with two tools:
1. `busqueda_libros_deep_learning` — local ChromaDB retriever with a **two-stage pipeline**:
   - Stage 1 (MMR): fetches 50 candidates, returns 20 diverse ones.
   - Stage 2 (cross-encoder reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`): re-scores all 20 candidates with full query↔document attention and keeps the top 4. This is the highest-ROI retrieval improvement per 2026 RAG benchmarks.
2. `busqueda_arxiv_papers` — ArXiv fallback for recent or missing topics.

`app.py` uses `importlib` to load `03_agente/agent.py` because Python cannot import modules whose names start with digits using standard `import`.

### Query expansion

`rag_utils/query_expansion.py` detects Spanish Deep Learning terms and appends English technical equivalents (e.g., "regularizacion" → "overfitting, regularization, dropout, weight decay") so queries in Spanish retrieve English-language papers correctly.

### Local artifacts (not committed)

`vector_db/`, `.python_packages/`, `.conda_packages_py39/`, `.chroma_onnx_cache/` are regenerable and excluded from git.

## Pilar 04 — Evaluación y Visualización (Fernanda)

### Objetivo
Medir la calidad del sistema RAG con métricas RAGAS y mostrar resultados
en una interfaz Streamlit.

### Archivos principales
- `04_evaluacion/evaluar_cobertura_direct.py` → cobertura del corpus
- `04_evaluacion/evaluar_profundidad_chunks.py` → profundidad de chunks
- `04_evaluacion/ragas_eval.py` → evaluación completa con RAGAS (en desarrollo)
- `app.py` → interfaz Streamlit del chatbot

### Comandos de evaluación
```bash
# Cobertura del corpus
python 04_evaluacion/evaluar_cobertura_direct.py --embedding-backend onnx-minilm -k 5

# Profundidad de chunks
python 04_evaluacion/evaluar_profundidad_chunks.py --embedding-backend onnx-minilm -k 5

# App Streamlit
streamlit run app.py
```

### Métricas RAGAS a implementar
- `faithfulness` → respuesta fiel al contexto recuperado
- `answer_relevancy` → relevancia de la respuesta a la pregunta
- `context_precision` → precisión del contexto recuperado
- `context_recall` → cobertura del contexto relevante

### Dependencias del pilar
Requiere que el vector_db/ ya esté construido (corre primero el Pilar 01).