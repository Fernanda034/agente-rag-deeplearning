import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from langchain_chroma import Chroma
from langchain_core.tools import Tool
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_groq import ChatGroq
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from rag_utils.onnx_embeddings import ChromaONNXEmbeddings
from rag_utils.query_expansion import expand_query_for_english_corpus

load_dotenv()

_DB_CANDIDATES = [
    os.getenv("CHROMA_DB_PATH", ""),
    "./vector_db",
    "./02_vectorstore/chroma_db",
]
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "deeplearning_corpus")

def _resolver_ruta_db() -> str:
    for path in _DB_CANDIDATES:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError("No se encontró la base de datos vectorial.")

def get_agent():
    # 1. Embeddings — mismo modelo ONNX que usó Camilo en la ingesta
    embeddings = ChromaONNXEmbeddings()

    # 2. Vector Store
    db_path = _resolver_ruta_db()
    print(f" Usando ChromaDB en: {db_path}")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=db_path,
        embedding_function=embeddings,
    )

    # 3. Retriever MMR
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20},
    )

    # 4. Tool 1: búsqueda local con query expansion
    def buscar_con_expansion(query: str) -> str:
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
        description="Busca información de libros y papers locales de Deep Learning. Úsalo SIEMPRE primero.",
    )

    # 5. Tool 2: ArXiv
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    arxiv_tool.name = "busqueda_arxiv_papers"
    arxiv_tool.description = "Busca papers recientes en arXiv. Úsalo solo si no encontraste en los libros locales."

    tools = [retriever_tool, arxiv_tool]

    # 6. LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

    # 7. Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente académico experto en Machine Learning y Deep Learning.

Instrucciones obligatorias:
1. Usa SIEMPRE 'busqueda_libros_deep_learning' primero.
2. Si no encuentras, usa 'busqueda_arxiv_papers'.
3. Fórmulas matemáticas en LaTeX entre signos de dólar: $formula$.
4. SIEMPRE cita la fuente: libro y página, o paper de arXiv.
5. Si no sabes la respuesta, admítelo claramente.
6. Responde en español, de forma clara y didáctica.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 8. Agente
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    return agent_executor
