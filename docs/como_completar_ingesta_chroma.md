# Como reproducir la ingesta con ChromaDB

## Estado actual

La ingesta directa con ChromaDB ya fue ejecutada y dejo evidencia en:

```text
docs/reporte_ingesta_chroma_direct.md
docs/tabla_alcance_agente.md
```

La base local existe en:

```text
vector_db/
```

Esa carpeta esta ignorada por Git porque es un artefacto regenerable.

## Preview de preprocesamiento

Para revisar extraccion, limpieza y chunks estimados:

```bash
python 01_ingesta/preprocess_preview.py
```

Genera:

```text
docs/reporte_preprocesamiento_preview.md
```

## Ingesta directa recomendada

Para crear o regenerar `vector_db/`:

```bash
python 01_ingesta/ingestion_chroma_direct.py --embedding-backend onnx-minilm
```

Resultado esperado:

```text
vector_db/
docs/reporte_ingesta_chroma_direct.md
```

## Entorno recomendado

Usar Python 3.10 o 3.11. Si se usa Conda:

```bash
conda create -n rag-dl python=3.11 -y
conda activate rag-dl
```

Entrar al proyecto:

```bash
cd "C:\Users\camil\OneDrive\Documentos\New project 6"
```

Instalar dependencias:

```bash
python -m pip install -r requirements_chroma_py39.txt
```

Verificar:

```bash
python -c "import chromadb, pypdf, sentence_transformers; print('ok')"
```

## Busqueda

```bash
python 02_vectorstore/retriever_chroma_direct.py "Que es Batch Normalization?" --topic regularizacion_estabilidad
```

## Evaluacion de alcance

```bash
python 04_evaluacion/evaluar_cobertura_direct.py --embedding-backend onnx-minilm -k 5
```

Genera:

```text
docs/tabla_alcance_agente.md
```

## Nota de Windows

Si el Python embebido de una herramienta falla con dependencias nativas, usar un entorno normal de Conda o Python instalado en el sistema. En este proyecto `vector_db/` es local y puede regenerarse sin afectar los archivos versionados.

## Fuentes consultadas

- Conda environments: https://docs.conda.io/en/latest/user-guide/tasks/manage-environments.html
- Conda Python versions: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html
- Chroma clients / PersistentClient: https://cookbook.chromadb.dev/core/clients/
