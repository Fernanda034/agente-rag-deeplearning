import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from rag_utils.onnx_embeddings import ChromaONNXEmbeddings

load_dotenv()

DB_DIR = ROOT_DIR / "vector_db"


def get_agent():
    """
    Construye y retorna el agente con herramientas (Retriever + reranker cross-encoder + ArXiv).
    """
    # 1. Cargar la base de datos vectorial con el mismo backend de embeddings que la ingesta
    embeddings = ChromaONNXEmbeddings()

    db_path = str(DB_DIR)
    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"No se encontró la base de datos en {db_path}. Corre el script de ingesta primero."
        )

    vectorstore = Chroma(
        collection_name="deeplearning_corpus",
        persist_directory=db_path,
        embedding_function=embeddings,
    )

    # 2. Retriever base con MMR — fetch amplio de candidatos para que el reranker filtre
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 20, "fetch_k": 50},
    )

    # 3. Reranker cross-encoder: re-puntúa los 20 candidatos con atención completa query↔doc
    #    y devuelve los 4 más relevantes. Mejora calidad ~15-30% vs. solo MMR (RAGAS benchmark 2026).
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=4)
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    # 4. Definir las Herramientas (Tools)
    retriever_tool = create_retriever_tool(
        retriever,
        name="busqueda_libros_deep_learning",
        description="Busca y retorna información de los libros y papers locales de Machine Learning y Deep Learning. Úsalo siempre como primera opción para contestar preguntas de la materia.",
    )

    arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000))
    arxiv_tool.name = "busqueda_arxiv_papers"
    arxiv_tool.description = "Busca papers académicos recientes en arXiv. Úsalo SÓLO si la información no se encontró en la herramienta de libros locales o si se pregunta por algoritmos muy nuevos."

    tools = [retriever_tool, arxiv_tool]

    # 5. Configurar el LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )

    # 6. Definir el Prompt del Agente
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente académico experto en Machine Learning y Deep Learning, diseñado para ayudar a estudiantes con sus dudas sobre la materia.

Instrucciones obligatorias:
1. Siempre intenta responder utilizando la herramienta de libros locales ('busqueda_libros_deep_learning') primero.
2. Si la información no está ahí, o si la pregunta es sobre un paper muy específico que no conoces, utiliza la herramienta de arXiv ('busqueda_arxiv_papers').
3. Si la respuesta requiere una fórmula matemática, escríbela en formato LaTeX rodeada por signos de dólar (por ejemplo: $w = \sum \alpha_i x_i$).
4. Siempre debes CITAR de dónde sacaste la información. Si usaste un libro, menciona el extracto. Si usaste arXiv, menciona el título del paper o ID.
5. Si no sabes la respuesta después de usar las herramientas, admite que no lo sabes.
6. Responde siempre en español de manera clara, didáctica y estructurada.
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 7. Crear el Agente
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 8. Ejecutor del Agente
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    return agent_executor
