# 📖 Guía de Uso Rápido: Ingesta y Evaluación

Esta guía está diseñada para que cualquier miembro del equipo pueda actualizar la base de datos de conocimiento y probar que el motor de búsqueda semántica (Retrieval) está funcionando correctamente antes de conectar a la Inteligencia Artificial.

---

## 🛠️ 1. ¿Cómo usar el script de Ingesta? (`01_ingesta/ingestion.py`)

**Propósito:** Este script lee los PDFs académicos, limpia el texto de basura (como guiones y símbolos raros), divide el texto en pequeños párrafos (chunks) y los convierte en vectores matemáticos que se guardan en la base de datos local (`vector_db`).

### Pasos para usarlo:
1. **Añade tus libros:** Si tienes un nuevo PDF de Machine Learning o Deep Learning, simplemente cópialo y pégalo dentro de la carpeta `corpus/`.
2. **Abre tu terminal** y asegúrate de estar en la carpeta raíz del proyecto (`agente-rag-deeplearning`).
3. **Activa tu entorno virtual** (si estás usando conda: `conda activate tu_entorno`).
4. **Ejecuta el script:**
   ```bash
   python 01_ingesta/ingestion.py
   ```
5. **Observa la magia:** El script hará un escaneo inteligente. Gracias a nuestro **Indexador Incremental**, si el PDF ya existía en la base de datos, lo ignorará para ahorrar tiempo. Si es un PDF nuevo, solo procesará y guardará ese archivo.

> [!TIP]
> **Atención:** Si agregas muchos libros, la primera vez puede tardar varios minutos. No cierres la consola hasta que veas el mensaje *"¡Ingesta completada con éxito!"*

---

## 🧪 2. ¿Cómo usar el Evaluador de Búsqueda? (`02_vectorstore/test_retrieval.py`)

**Propósito:** Este script es una consola interactiva (estilo terminal) que te permite hacer búsquedas matemáticas directas en la base de datos *sin usar a Llama 3 para responder*. Es la mejor herramienta para probar si la Ingesta funcionó bien y si los vectores están trayendo los párrafos correctos.

### Pasos para usarlo:
1. **Abre tu terminal** en la carpeta raíz del proyecto.
2. **Ejecuta el script de evaluación:**
   ```bash
   python 02_vectorstore/test_retrieval.py
   ```
3. **Interactúa:** La consola te pedirá que ingreses un concepto (Ejemplo: *"Backpropagation"*, *"Gradient Descent"*, o *"Support Vector Machines"*).
4. **Revisa los resultados:** El sistema utilizará la técnica **Multi-Query Expansion**. Es decir, Llama 3 reescribirá tu pregunta de 3 formas distintas en secreto, buscará en los vectores y te imprimirá en pantalla los **fragmentos exactos extraídos de los libros**, diciéndote en qué página lo encontró.
5. **Para salir:** Simplemente escribe `salir` y presiona Enter.

> [!NOTE]
> **¿Para qué sirve esto en el mundo real?** Si en la página web principal (`app.py`) el Chatbot te da una respuesta alucinada o incorrecta, tu primer paso de "Debugging" debe ser correr este script de evaluación. Si este script NO encuentra el párrafo correcto, el error está en la base de datos (Ingesta). Si este script SÍ encuentra el párrafo correcto, el error es que Llama 3 no entendió el contexto en el `agent.py`.
