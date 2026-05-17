# Arquitectura RAG con metadatos academicos

## Objetivo

Organizar el agente RAG para que no procese los PDFs como una bolsa unica de documentos, sino como un corpus academico dividido por ramas, temas y subtemas.

## Ramas actuales

| Rama | Uso |
| --- | --- |
| `deep_learning` | Rama principal del proyecto y de la presentacion. |
| `machine_learning_estadistica` | Rama secundaria con libros de ML clasico y Statistical Learning. Se usa solo como apoyo. |

## Flujo

```text
PDFs del corpus
  ->
Manifest de metadatos
  ->
Ingesta con limpieza y chunking
  ->
Chunks con branch/topic/subtopic/page/source
  ->
Embeddings semanticos locales ONNX MiniLM
  ->
ChromaDB
  ->
Pregunta del usuario + expansion bilingue si pregunta en espanol
  ->
Retriever con filtros por rama/tema/subtema
  ->
Respuesta del agente con fuente y pagina
```

## Archivos creados

| Archivo | Funcion |
| --- | --- |
| `docs/metadatos_temas_corpus.csv` | Define rama, tema, subtema y uso de cada fuente. |
| `01_ingesta/ingestion_chroma_direct.py` | Lee PDFs, limpia texto, crea chunks, calcula embeddings semanticos y guarda metadatos en ChromaDB. |
| `02_vectorstore/retriever_chroma_direct.py` | Permite buscar en ChromaDB filtrando por rama, tema o subtema. |
| `rag_utils/query_expansion.py` | Enriquece preguntas en espanol con equivalentes tecnicos en ingles para consultar papers en ingles. |
| `04_evaluacion/evaluar_cobertura_direct.py` | Genera una tabla de alcance por tema usando retrieval semantico. |
| `docs/guia_metadatos_corpus.md` | Explica los campos del manifest y como usarlos. |

## Donde poner PDFs

Estructura sugerida:

```text
corpus/
  ciencia_datos/
    deep_learning/
      05_atencion_transformers/
      06_cnn_vision/
      07_explicabilidad/
      08_autoencoders_vae_gans/
    machine_learning_estadistica/
```

Si se mantiene la carpeta plana `corpus/` de GitHub, tambien funciona, siempre que el archivo este registrado en el manifest.

## Como ejecutar ingesta

```bash
python 01_ingesta/ingestion_chroma_direct.py --embedding-backend onnx-minilm
```

Opcional:

```bash
python 01_ingesta/ingestion_chroma_direct.py --embedding-backend onnx-minilm --chunk-size 1000 --chunk-overlap 200
```

Resultado esperado:

- ChromaDB local en `vector_db/`.
- Reporte por tema en `docs/reporte_ingesta_chroma_direct.md`.

## Como probar retrieval por tema

Ejemplo para ResNet:

```bash
python 02_vectorstore/retriever_chroma_direct.py "Por que ResNet usa conexiones residuales?" --topic cnn_vision
```

Ejemplo para Transformers:

```bash
python 02_vectorstore/retriever_chroma_direct.py "Que es self-attention?" --topic atencion_transformers
```

## Como generar tabla de alcance

```bash
python 04_evaluacion/evaluar_cobertura_direct.py --embedding-backend onnx-minilm -k 5
```

Salida:

```text
docs/tabla_alcance_agente.md
```

## Argumento para la presentacion

> La mejora principal es que cada chunk conserva metadatos academicos: rama, tema, subtema, fuente y pagina. Ademas, la recuperacion usa embeddings semanticos locales con `all-MiniLM-L6-v2`, no busqueda literal por palabras. Como muchos papers estan en ingles y el usuario puede preguntar en espanol, el retriever enriquece la pregunta con equivalentes tecnicos en ingles antes de buscar. Esto permite filtrar por tema y recuperar fragmentos conceptualmente cercanos con fuente y pagina.

## Siguiente paso

Integrar el retriever al agente final para que el LLM use los chunks recuperados y cite fuente/pagina en la respuesta.
