# Agente RAG — Deep Learning & Machine Learning

Asistente académico inteligente basado en RAG (Retrieval-Augmented Generation) especializado en Deep Learning y Machine Learning. Responde preguntas en español con fuentes citadas, fórmulas LaTeX y referencias a papers.

**Demo en vivo:** https://agente-rag-deeplearning.vercel.app

---

## Arquitectura general

```
corpus/ PDFs
  → 01_ingesta/          extracción, limpieza, chunking → ChromaDB
  → 02_vectorstore/      recuperación filtrada por tema/subtema
  → rag_utils/           expansión de consultas ES→EN, embeddings ONNX
  → 03_agente/           agente ReAct (Groq Llama 3.3 70B + ArXiv)
  → api/chat.py          Flask serverless (Vercel)
  → public/index.html    interfaz web (Tailwind + KaTeX + marked.js)
```

El sistema opera en **dos modos automáticos**:
- **RAG** — cuando `vector_db/` existe localmente: recupera chunks del corpus con ChromaDB y reranking.
- **Groq directo** — en Vercel (sin `vector_db/`): usa un system prompt experto y Llama 3.3 70B directamente.

---

## Equipo y pilares

| Pilar | Responsable | Descripción |
|-------|-------------|-------------|
| `01_ingesta` | Juan Camilo | Extracción PDF, limpieza regex, chunking, ingesta ChromaDB con metadatos |
| `02_vectorstore` | Juan Camilo / María | Retriever con filtros por `branch/topic/subtopic`, evaluación de cobertura |
| `03_agente` | María | Agente ReAct con herramienta RAG + fallback ArXiv, cross-encoder reranker |
| `04_evaluacion` + deploy | Fernanda | Evaluación RAGAS, interfaz web, despliegue Vercel |

---

## Uso local (modo RAG completo)

### 1. Requisitos

Python 3.10 o 3.11. Crear `.env` desde el ejemplo:

```bash
cp .env.example .env
# Editar .env y agregar: GROQ_API_KEY=gsk_...
```

Instalar dependencias:

```bash
pip install -r requirements_chroma_py39.txt   # embeddings + ChromaDB
pip install -r requirements.txt               # LangChain + Groq + RAGAS
```

### 2. Construir la base vectorial

```bash
python 01_ingesta/ingestion_chroma_direct.py --embedding-backend onnx-minilm
```

### 3. Levantar el servidor local

```bash
python api/chat.py
# Abre: http://localhost:8000
```

---

## Evaluación

```bash
# Cobertura del corpus por tema
python 04_evaluacion/evaluar_cobertura_direct.py --embedding-backend onnx-minilm -k 5

# Profundidad de chunks
python 04_evaluacion/evaluar_profundidad_chunks.py --embedding-backend onnx-minilm -k 5

# Evaluación RAGAS completa (requiere vector_db/ y GROQ_API_KEY)
python 04_evaluacion/ragas_eval.py --k 4
# Salida: docs/ragas_report.md
```

---

## Despliegue (Vercel)

La app está desplegada en https://agente-rag-deeplearning.vercel.app y funciona sin `vector_db/` gracias al modo Groq directo.

Para redesplegar:

```bash
vercel --prod
```

Variable de entorno requerida en el dashboard de Vercel: `GROQ_API_KEY`.

---

## Artefactos locales (no versionados)

`vector_db/`, `.chroma_onnx_cache/`, `.python_packages/` son regenerables y están en `.gitignore`.
