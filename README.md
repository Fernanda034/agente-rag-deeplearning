# agente-rag-deeplearning
Agente inteligente basado en modelos de lenguaje (LLMs) especializado en el dominio de Deep Learning

## Aporte de Juan Camilo - ingesta y corpus RAG

Esta rama agrega la parte de ingesta y preparacion del corpus academico de Deep Learning para que el agente RAG no consulte PDFs como una bolsa unica, sino por rama, tema, subtema, fuente y pagina.

### Que se hizo

- Organizacion del corpus por temas de Deep Learning en `corpus/ciencia_datos/deep_learning/`.
- Registro de metadatos academicos en `docs/metadatos_temas_corpus.csv`.
- Extraccion de texto de PDFs pagina por pagina.
- Limpieza basica de texto extraido y division en chunks con overlap.
- Ingesta directa en ChromaDB con embeddings locales ONNX MiniLM.
- Retriever con filtros por `branch`, `topic` y `subtopic`.
- Expansion bilingue de consultas en espanol hacia terminos tecnicos en ingles.
- Evaluacion basica de alcance y evaluacion profunda de chunks.

### Archivos principales

```text
01_ingesta/
  preprocess_preview.py
  ingestion_chroma_direct.py
02_vectorstore/
  retriever_chroma_direct.py
04_evaluacion/
  evaluar_cobertura_direct.py
  evaluar_profundidad_chunks.py
rag_utils/
  query_expansion.py
docs/
  arquitectura_rag_metadatos.md
  metadatos_temas_corpus.csv
  reporte_preprocesamiento_preview.md
  reporte_ingesta_chroma_direct.md
  tabla_alcance_agente.md
  evaluacion_profundidad_chunks.md
```

### Como revisar el trabajo

Instalar dependencias en Python 3.10 o 3.11:

```bash
python -m pip install -r requirements_chroma_py39.txt
```

Generar preview de preprocesamiento:

```bash
python 01_ingesta/preprocess_preview.py
```

Regenerar la base vectorial local:

```bash
python 01_ingesta/ingestion_chroma_direct.py --embedding-backend onnx-minilm
```

Probar busqueda por tema:

```bash
python 02_vectorstore/retriever_chroma_direct.py "Que es Batch Normalization?" --topic regularizacion_estabilidad
```

Generar evaluacion de cobertura:

```bash
python 04_evaluacion/evaluar_cobertura_direct.py --embedding-backend onnx-minilm -k 5
```

Generar evaluacion profunda:

```bash
python 04_evaluacion/evaluar_profundidad_chunks.py --embedding-backend onnx-minilm -k 5
```

### Despliegue en Vercel

Este proyecto está preparado para desplegarse en Vercel con Python. Se agregó `vercel.json` y `runtime.txt` para que Vercel use `app.py` como entrada.

Antes de desplegar, asegúrate de configurar en Vercel la variable de entorno:

- `GROQ_API_KEY`

Si no está definida, la app mostrará un error en tiempo de ejecución.

### Alcance actual

La evaluacion profunda queda documentada en `docs/evaluacion_profundidad_chunks.md`. El corpus queda fuerte en atencion/Transformers, Adam, regularizacion/estabilidad, VAE/GAN, ResNet y Grad-CAM. Los huecos declarados para una siguiente iteracion son LSTM con fuente local directa, CNN basica/transfer learning, autoencoders basicos, weight decay/early stopping/inicializacion y saliency maps generales.

`vector_db/`, `.python_packages/` y `.chroma_onnx_cache/` no se suben a GitHub porque son artefactos locales regenerables.
