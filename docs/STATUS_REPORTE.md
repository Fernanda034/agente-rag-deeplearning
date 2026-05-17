# 🚀 Reporte de Avance: Agente Inteligente RAG para Deep Learning

**Fecha:** 16 de Mayo de 2026  
**Responsable del avance actual:** Fernanda  
**Estado General:** Fase de Arquitectura de Datos Completada (Backend & Ingesta listos). Interfaz lista para despliegue.

---

## 1. 🎯 Lo que se ha logrado (Completado)

Hemos construido la columna vertebral del proyecto, superando los estándares básicos del RAG tradicional:

* **Ingesta Avanzada de Datos (`01_ingesta`):**
  * No nos conformamos con un lector simple. Implementamos limpieza exhaustiva con Expresiones Regulares (Regex) para eliminar "basura" de los PDFs académicos (cortes de palabras por guiones, saltos de línea intermedios, caracteres Unicode subrogados).
  * **Indexación Incremental:** Se integró un `SQLRecordManager` que le saca un *hash* (huella digital) a cada fragmento. Si agregamos o borramos PDFs en el futuro, el sistema solo actualiza los cambios sin reprocesar todo desde cero.
* **Base de Datos Vectorial (`02_vectorstore`):**
  * Los PDFs (incluyendo los de Deep Learning y agregados de Machine Learning Clásico) fueron particionados y vectorizados exitosamente usando ChromaDB en formato local. Tenemos más de 10,000 fragmentos "puros" listos.
  * Se creó el script de evaluación `test_retrieval.py` que comprueba matemáticamente que el sistema extrae las páginas y textos correctos ante una pregunta.
* **Agente ReAct Inteligente (`03_agente`):**
  * Se configuró el cerebro del bot usando **Llama 3 (70B) vía Groq** por su velocidad extrema.
  * **Múltiples Herramientas:** El agente es bilingüe y toma decisiones. Primero busca en los libros locales (RAG), y si le preguntan por algoritmos de ayer, tiene la capacidad de conectarse a la API de **ArXiv** para buscar papers en tiempo real.

---

## 2. 🔄 Desviaciones respecto a la propuesta original (Consenso)

Al comparar nuestro progreso con el documento `Propuesta_Agente_RAG_DeepLearning.docx`, tomamos un par de decisiones técnicas para mejorar la calidad y ajustarnos a los tiempos:

1. **Ampliación del Corpus:** La propuesta se limitaba estrictamente a Deep Learning. Hemos agregado literatura fundamental de Machine Learning (*Introduction to Statistical Learning*) para darle más versatilidad al bot si le preguntan conceptos base (ej. Regresión vs Clasificación).
2. **Motor de Inferencia (Groq vs OpenAI):** En lugar de usar modelos lentos o de pago, estamos utilizando las LPUs de **Groq** con Llama 3. Esto nos da tiempos de respuesta de milisegundos, ideal para una demo en vivo.
3. **Base de Datos Local (ChromaDB):** Para asegurar tener un entregable funcional en una semana, decidimos mantener la base de datos en formato local (archivos `.bin`) en lugar de subirla a la nube (Supabase/Pinecone). Esto es más fácil de presentar y no requiere configuraciones de red complejas.

---

## 3. 🚧 Lo que falta (Próximos pasos para el equipo)

1. **Evaluación de la Interfaz Web:** Levantar `app.py` en Streamlit y hacer pruebas de "estrés" preguntándole cosas difíciles al chat.
2. **Despliegue (Opcional pero recomendado):** Subir el código a Streamlit Community Cloud para que los profesores puedan entrar desde su celular. *(Nota: La carpeta `vector_db` deberá subirse a GitHub si pesa menos de 100MB, de lo contrario hay que usar Pinecone).*

## 4. 🏗️ Deuda Técnica y Mejoras Arquitectónicas (Backlog del Equipo)

A medida que el proyecto crezca, el equipo debería considerar implementar estas mejoras para llegar al "Estado del Arte" (SOTA):

1. **Refactorización de `02_vectorstore` (Single Responsibility):** Actualmente, el archivo `agent.py` se encarga de inicializar Chroma y los embeddings. Para mantener una arquitectura limpia, toda la lógica de conexión a la base de datos vectorial debería mudarse a un archivo dedicado (ej. `02_vectorstore/database.py`), para que el Agente solo la importe y se dedique a pensar.
2. **Nuevas Herramientas (Tools) para Llama 3:**
    * *Python REPL:* Permitir al agente hacer cálculos matemáticos exactos (ej. cálculo de tensores y parámetros).
    * *Búsqueda Web:* Conectar a Tavily/DuckDuckGo para resolver dudas sobre papers o frameworks que salieron la última semana.
3. **Técnicas RAG Ultra-Avanzadas:** Implementar "Re-Ranking" (ej. con Cohere) para ordenar mejor los vectores recuperados antes de pasarlos al LLM, o un flujo de "Self-RAG" donde el bot evalúe la pertinencia de la info antes de responder.
4. **Migración a Pinecone:** Para habilitar un flujo de trabajo 100% colaborativo sin tener que lidiar con Git LFS ni bases de datos locales pesadas. *(Ver el documento `Propuesta_Migracion_Pinecone.docx` adjunto).*
