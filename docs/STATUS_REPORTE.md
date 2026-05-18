# Reporte de Estado: Agente Inteligente RAG para Deep Learning

**Fecha de última actualización:** 17 de Mayo de 2026
**Estado General:** ✅ Proyecto completado y desplegado en producción

**URL de producción:** https://agente-rag-deeplearning.vercel.app

---

## 1. Lo que se ha completado

### Pilar 01 — Ingesta (Juan Camilo)
- Limpieza exhaustiva de PDFs con regex (guiones, saltos de línea, Unicode)
- Chunking con overlap y metadatos por `branch / topic / subtopic / chapter / page`
- Ingesta directa en ChromaDB con embeddings ONNX MiniLM-L6-v2
- Script de indexación incremental (`SQLRecordManager`) — solo reprocesa cambios
- Expansión bilingüe de consultas ES→EN (`rag_utils/query_expansion.py`)
- **Resultado:** >10 000 fragmentos indexados en `vector_db/`

### Pilar 02 — Vectorstore (Juan Camilo / María)
- Retriever con filtros por `branch`, `topic` y `subtopic`
- Script de evaluación de cobertura por tema (`evaluar_cobertura_direct.py`)
- Script de evaluación de profundidad de chunks (`evaluar_profundidad_chunks.py`)
- Corrección de mismatch de embeddings: `ChromaONNXEmbeddings` garantiza que query y documentos usen el mismo modelo ONNX end-to-end

### Pilar 03 — Agente (María)
- Agente ReAct con Groq Llama 3.3 70B (herramienta RAG + fallback ArXiv)
- Pipeline de recuperación en dos etapas: MMR (50→20 candidatos) + cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`, top 4)
- Corrección de imports `langchain_classic` → `langchain` actual

### Pilar 04 — Evaluación e Interfaz (Fernanda)
- Script RAGAS completo (`04_evaluacion/ragas_eval.py`): 7 preguntas con ground truth, métricas `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- Interfaz web (`public/index.html`): Tailwind CSS, KaTeX, marked.js v9, chips de fuentes RAG, badge de modo
- API Flask serverless dual-mode (`api/chat.py` + `api/health.py`)
- **Despliegue en Vercel:** https://agente-rag-deeplearning.vercel.app ✅ en producción

---

## 2. Desviaciones respecto a la propuesta original

| Decisión | Propuesta original | Implementación final | Motivo |
|----------|-------------------|---------------------|--------|
| Corpus | Solo Deep Learning | DL + ML Estadística | Mayor cobertura para preguntas base |
| LLM | OpenAI GPT | Groq Llama 3.3 70B | Velocidad (LPUs) y costo cero |
| Vector DB | Pinecone/Supabase | ChromaDB local + Vercel sin DB | Entregable funcional sin config de red |
| Frontend | Streamlit | HTML + Tailwind + Vercel | Deploy serverless sin límites de Streamlit Cloud |
| Retrieval | MMR básico | MMR + cross-encoder reranker | Mejor precisión (+SOTA 2026) |

---

## 3. Pendiente para completar la entrega

| Tarea | Responsable | Notas |
|-------|-------------|-------|
| Ejecutar `ragas_eval.py` y subir `docs/ragas_report.md` | Fernanda | Requiere `vector_db/` local y `GROQ_API_KEY` en `.env` |

---

## 4. Backlog técnico (post-entrega)

- **Migración a Pinecone:** flujo 100% cloud sin Git LFS (ver `docs/Propuesta_Migracion_Pinecone.docx`)
- **Python REPL tool:** permitir al agente calcular parámetros de redes exactamente
- **Búsqueda web (Tavily):** complementar ArXiv con papers de la última semana
- **Huecos en corpus:** LSTM local, CNN básica, autoencoders, weight decay/early stopping, saliency maps
