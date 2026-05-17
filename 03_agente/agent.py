"""
03_agente Modelado e Inteligencia

Cambios respecto al código original de Codex:
  [FIX] langchain_classic no existe → corregido a langchain.agents
  [FIX] Ruta vector_db unificada con variable de entorno
  [ADD] query_expansion integrado en el retriever tool
  [ADD] Comentarios explicativos en cada sección
────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
from dotenv import load_dotenv

# ── Asegurar que rag_utils sea importable desde cualquier directorio ───────────
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import create_retriever_tool, Tool
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_groq import ChatGroq

# import langchain.agents
from langchain.agents import create_tool_calling_agent, AgentExecutor

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# query_expansion que Camilo dejó sin conectar
from rag_utils.query_expansion import expand_query_for_english_corpus

load_dotenv()

# ── Configuración desde .env ───────────────────────────────────────────────────
# Busca la DB en la ruta del .env; si no existe, prueba rutas alternativas
_DB_CANDIDATES = [
    os.getenv("CHROMA_DB_PATH", ""),          # desde .env (prioritario)
    "./vector_db",                              # ruta original de Camilo
    "./02_vectorstore/chroma_db",              # ruta estándar del proyecto
]
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "deeplearning_corpus")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _resolver_ruta_db() -> str:
    """Encuentra la primera ruta válida donde existe la base de datos."""
    for path in _DB_CANDIDATES:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No se encontró la base de datos vectorial en ninguna ruta conocida.\n"
        "Rutas buscadas:\n" + "\n".join(f"  - {p}" for p in _DB_CANDIDATES if p) +
        "\n\nSolución: corre el script de ingesta de Mafe/Camilo primero, "
        "o configura CHROMA_DB_PATH en tu .env"
    )


def get_agent():
    """
    Construye y retorna el AgentExecutor listo para usar.
    Llamado una sola vez por app.py y guardado en session_state.
    """

    # ── 1. Embeddings ──────────────────────────────────────────────────────────
    # Mismo modelo que usó Mafe para crear ChromaDB — deben coincidir
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── 2. Vector Store ────────────────────────────────────────────────────────
    db_path = _resolver_ruta_db()
    print(f" Usando ChromaDB en: {db_path}")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=db_path,
        embedding_function=embeddings,
    )

    # ── 3. Retriever con MMR ───────────────────────────────────────────────────
    # MMR (Maximal Marginal Relevance): devuelve chunks relevantes pero diversos
    # fetch_k=20 → trae 20 candidatos, luego selecciona los 4 más diversos
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20},
    )

    # ── 4. Tool 1: Búsqueda local con query expansion ──────────────────────────
    # INTEGRACIÓN: expandimos la query antes de buscar para manejar
    # preguntas en español sobre un corpus en inglés.
    # Ejemplo: "¿qué es la atención?" → agrega "attention, self-attention, transformer"

    def buscar_con_expansion(query: str) -> str:
        """Wrapper que expande la query antes de buscar en ChromaDB."""
        query_expandida = expand_query_for_english_corpus(query)
        docs = retriever.invoke(query_expandida)
        if not docs:
            return "No se encontró información relevante en los documentos locales."
        resultados = []
        for doc in docs:
            fuente = doc.metadata.get("source", "fuente desconocida")
            pagina = doc.metadata.get("page", "?")
            resultados.append(f"[{fuente}, p.{pagina}]\n{doc.page_content}")
        return "\n\n---\n\n".join(resultados)

    retriever_tool = Tool(
        name="busqueda_libros_deep_learning",
        func=buscar_con_expansion,
        description=(
            "Busca y retorna información de los libros y papers locales de "
            "Machine Learning y Deep Learning. Úsalo SIEMPRE como primera opción "
            "para responder preguntas de la materia. Incluye el nombre del libro "
            "y la página en la respuesta."
        ),
    )

    # ── 5. Tool 2: ArXiv para papers recientes ─────────────────────────────────
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    arxiv_tool.name = "busqueda_arxiv_papers"
    arxiv_tool.description = (
        "Busca papers académicos recientes en arXiv. Úsalo SOLO si la información "
        "no se encontró en los libros locales, o si se pregunta por técnicas muy recientes."
    )

    tools = [retriever_tool, arxiv_tool]

    # ── 6. LLM ─────────────────────────────────────────────────────────────────
    # llama-3.3-70b-versatile: más capaz que el 8b, sigue siendo gratis en Groq
    # temperature=0.2: un poco más de fluidez que 0, pero aún muy preciso
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )

    # ── 7. Prompt del agente ───────────────────────────────────────────────────
    # agent_scratchpad es donde LangChain registra el "pensamiento" del agente
    # (qué tool usó, qué devolvió, qué va a responder)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente académico experto en Machine Learning y Deep Learning, \
diseñado para ayudar a estudiantes con sus dudas sobre la materia.

Instrucciones obligatorias:
1. Usa SIEMPRE la herramienta 'busqueda_libros_deep_learning' primero.
2. Si no encuentras la respuesta ahí, usa 'busqueda_arxiv_papers' para papers recientes.
3. Si la respuesta requiere una fórmula matemática, escríbela en LaTeX entre signos de dólar \
   (ejemplo: $w = \\sum \\alpha_i x_i$).
4. SIEMPRE cita la fuente al final: libro y página, o título del paper de arXiv.
5. Si no sabes la respuesta tras usar las herramientas, admítelo claramente.
6. Responde siempre en español, de forma clara, didáctica y estructurada.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # ── 8. Construir y retornar el AgentExecutor ───────────────────────────────
    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,           # muestra el "pensamiento" en consola (útil para debug)
        handle_parsing_errors=True,
        max_iterations=5,       # evita loops infinitos
    )

    return agent_executor
