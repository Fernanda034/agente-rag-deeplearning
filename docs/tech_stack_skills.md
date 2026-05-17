# Documentación de Componentes y Librerías (Tech Stack Skills)

Este documento detalla cada una de las tecnologías que usaremos en su versión más actual (2024), cómo funcionan, y cómo las implementaremos para nuestro Agente Inteligente RAG.

## Estructura Recomendada del Repositorio

Para mantener el código ordenado y modular, esta es la estructura que implementaremos:

```text
agente-rag-deeplearning/
├── .env                  # Variables de entorno (API Keys) - NO se sube a GitHub
├── .gitignore
├── requirements.txt      # Dependencias del proyecto
├── app.py                # Interfaz principal de Streamlit
├── corpus/               # (Carpeta local) Aquí pones los PDFs de la materia
├── docs/                 # Documentación del proyecto (como este archivo)
├── vector_db/            # (Generado automáticamente) Base de datos local ChromaDB
└── src/                  # Lógica interna del agente
    ├── ingestion.py      # Script para leer PDFs, partirlos y guardarlos en Chroma
    ├── retriever.py      # Lógica de búsqueda semántica (MMR)
    ├── agent.py          # Definición del agente ReAct, prompts y herramientas
    └── evaluation.py     # Script para correr RAGAS y evaluar el modelo
```

---

## Componentes y Librerías

A continuación, la guía "tipo skills" para cada componente fundamental:

### 1. LangChain (Core)
* **¿Qué es?** Es el framework orquestador. Nos permite conectar el LLM (Groq) con los datos (Chroma) y darle capacidades de razonamiento (Agents & Tools).
* **Versión Actual:** LangChain modularizó todo en 2024. Ya no se usa un solo paquete gigante, sino paquetes pequeños y específicos.
* **Implementación:** Instalaremos `langchain`, `langchain-core` y `langchain-community`.
* **Concepto clave:** Usaremos **Chains** para flujos lineales y **Agents** para que el LLM decida si necesita buscar en el PDF o usar su conocimiento general.

### 2. Groq API (`langchain-groq`)
* **¿Qué es?** Groq es un proveedor de inferencia que corre modelos open-source (como Llama 3 8B y 70B) en unidades de procesamiento ultra-rápidas (LPUs). Ofrece un plan gratuito muy generoso.
* **Implementación:**
  1. Te crearás una cuenta en [console.groq.com](https://console.groq.com/).
  2. Generarás una API Key y la pondremos en el archivo `.env` como `GROQ_API_KEY=tu_clave_aqui`.
  3. En código, usaremos la integración oficial:
  ```python
  from langchain_groq import ChatGroq
  llm = ChatGroq(model="llama3-8b-8192") # Corre Llama 3!
  ```

### 3. ChromaDB (`langchain-chroma`)
* **¿Qué es?** Una base de datos vectorial open-source. Almacena texto convertido en números (embeddings) para poder buscar frases "por significado" y no por coincidencia exacta de palabras.
* **Implementación:** 
  - La instalamos junto a su paquete integrador: `chromadb` y `langchain-chroma`.
  - Cuando procesemos los PDFs del `corpus/`, Chroma guardará los resultados en una carpeta local llamada `vector_db/`. No necesitas servidores externos.
  ```python
  from langchain_chroma import Chroma
  from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

  # Usaremos embeddings gratuitos locales de HuggingFace
  embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
  vector_store = Chroma(persist_directory="./vector_db", embedding_function=embeddings)
  ```

### 4. PyPDF y Divisores de Texto
* **¿Qué es?** Antes de guardar el PDF en ChromaDB, hay que leerlo y partirlo en pedacitos.
* **Implementación:** Usaremos `PyPDFLoader` para leer el texto, y `RecursiveCharacterTextSplitter` para cortarlo.
  - *¿Por qué cortarlo?* El LLM tiene un límite de memoria. Si le pasamos un PDF de 500 páginas, colapsará. Lo cortamos en "chunks" de 1000 letras, y cuando preguntes algo, ChromaDB buscará solo los 4 chunks más relevantes y se los dará a Groq.

### 5. Streamlit (`streamlit`)
* **¿Qué es?** Librería para crear la interfaz web (UI) del chat escribiendo únicamente código Python. 
* **Implementación:** Usaremos los componentes modernos de chat lanzados en 2024: `st.chat_message` y `st.chat_input`. Todo el historial de chat se guardará temporalmente en `st.session_state`.
  ```python
  import streamlit as st

  st.title("Asistente de Deep Learning")
  prompt = st.chat_input("Hazme una pregunta sobre la materia...")
  
  if prompt:
      with st.chat_message("user"):
          st.write(prompt)
      # Aquí llamaríamos a nuestro Agente Groq y pintaríamos la respuesta
  ```

### 6. Evaluación con RAGAS (`ragas`)
* **¿Qué es?** Un framework para saber si tu chatbot RAG es bueno o malo matemáticamente. 
* **Implementación:** Crearemos un archivo `evaluation.py`. Le pasaremos preguntas pre-armadas y las respuestas que generó Groq. RAGAS usará un LLM (usualmente otro modelo) como "Juez" para darnos notas del 0 al 1 en métricas como:
  - **Faithfulness:** Si dijo la verdad basándose en el PDF o si inventó datos.
  - **Answer Relevance:** Si contestó a lo que le preguntamos.

### 7. Preparación Avanzada y Limpieza de Texto
* **¿Qué es?** Los PDFs académicos están llenos de "basura" (encabezados, pies de página, saltos de línea a mitad de oración) que confunden al LLM. Además, cortar el texto arbitrariamente cada 1000 letras puede romper conceptos.
* **Implementación Avanzada:** 
  - **Limpieza (Cleaning):** Antes de guardar el texto, usaremos expresiones regulares (Regex) para unir palabras cortadas por guiones (ej. "hiper- \n parámetro") y limpiar saltos de línea innecesarios.
  - **Metadatos:** Mantendremos el nombre del archivo fuente y la página para que el LLM pueda citarlo.

### 8. Indexación Incremental (RecordManager)
* **¿Qué es?** Si corres el script de ingesta de nuevo sobre la misma base de datos, `ChromaDB.from_documents` normalmente **duplicaría** todos los fragmentos. Si agregas un solo PDF nuevo, no deberías tener que reprocesar los 7 anteriores.
* **Implementación:** Usaremos el **LangChain Indexing API** junto con `SQLRecordManager`. 
  - Esta API crea una pequeña base de datos SQLite paralela que guarda un "hash" (una huella digital) de cada chunk. 
  - Al correr la ingesta, LangChain revisará: *¿Este PDF ya fue procesado y no ha cambiado?* Si es así, lo ignora. *¿Es un PDF nuevo?* Lo agrega. *¿Se borró un PDF de la carpeta?* Lo elimina de ChromaDB.
  ```python
  from langchain.indexes import SQLRecordManager, index
  record_manager = SQLRecordManager("chroma/deeplearning", db_url="sqlite:///record_manager.db")
  index(docs, record_manager, vectorstore, cleanup="incremental", source_id_key="source")
  ```

---

## Siguientes Pasos (Qué debes hacer tú)

1. Crear tu cuenta gratuita en **Groq** y obtener una API Key.
2. Reunir los 2 PDFs iniciales y ponerlos en la carpeta `corpus/`.
3. Avisarme cuando estés listo para empezar a codificar.
