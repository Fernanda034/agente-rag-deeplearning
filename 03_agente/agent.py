import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_groq import ChatGroq
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

DB_DIR = "../vector_db"

def get_agent():
    """
    Construye y retorna el agente ReAct con las herramientas (Retriever y ArXiv)
    """
    # 1. Cargar la base de datos vectorial
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Asegurarnos de que el path a vector_db es correcto cuando se corre desde app.py
    # app.py suele correrse en la raíz
    db_path = "./vector_db" if os.path.exists("./vector_db") else DB_DIR
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"No se encontró la base de datos en {db_path}. Corre el script de ingesta primero.")
        
    vectorstore = Chroma(
        collection_name="deeplearning_corpus",
        persist_directory=db_path,
        embedding_function=embeddings
    )
    
    # 2. Configurar el retriever con búsqueda MMR (diversidad + relevancia)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 4, 'fetch_k': 20}
    )
    
    # 3. Definir las Herramientas (Tools)
    retriever_tool = create_retriever_tool(
        retriever,
        name="busqueda_libros_deep_learning",
        description="Busca y retorna información de los libros y papers locales de Machine Learning y Deep Learning. Úsalo siempre como primera opción para contestar preguntas de la materia."
    )
    
    arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000))
    arxiv_tool.name = "busqueda_arxiv_papers"
    arxiv_tool.description = "Busca papers académicos recientes en arXiv. Úsalo SÓLO si la información no se encontró en la herramienta de libros locales o si se pregunta por algoritmos muy nuevos."
    
    tools = [retriever_tool, arxiv_tool]
    
    # 4. Configurar el LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # Usamos el modelo más capaz para reasoning
        temperature=0.2
    )
    
    # 5. Definir el Prompt del Agente
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
    
    # 6. Crear el Agente
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 7. Ejecutor del Agente
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor
