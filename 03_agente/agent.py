import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_core.tools import tool
import requests
from langchain_groq import ChatGroq
from langchain.agents import create_agent

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

DB_DIR = ROOT_DIR / "vector_db"


def get_agent():
    """
    Construye y retorna el agente con herramientas (Retriever + ArXiv + Semantic Scholar).
    """
    # 1. Cargar la base de datos vectorial en la nube (Pinecone) con embeddings de Hugging Face
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=os.environ.get("HF_TOKEN")
    )

    index_name = os.environ.get("PINECONE_INDEX_NAME", "deeplearning-corpus")

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.environ.get("PINECONE_API_KEY")
    )

    # 2. Retriever base con MMR (Pinecone) devolviendo los 4 más relevantes
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20},
    )

    # 3. Herramienta de Búsqueda con Expansión usando Runnable
    from langchain_core.runnables import RunnableLambda
    
    def expand_query(query: str) -> str:
        try:
            llm_exp = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
            prompt_exp = PromptTemplate.from_template(
                "Expande la siguiente consulta agregando sinónimos o conceptos relacionados "
                "para mejorar la búsqueda en una base de datos vectorial de Deep Learning. "
                "Devuelve SOLO las palabras clave separadas por espacio, sin explicaciones.\n\n"
                "Consulta original: {query}"
            )
            return (prompt_exp | llm_exp).invoke({"query": query}).content
        except Exception:
            return query
            
    # Conectamos la expansión de consulta directamente al retriever
    retriever_con_expansion = RunnableLambda(expand_query) | retriever
    
    # Creamos la herramienta usando la función oficial de LangChain
    busqueda_libros_deep_learning = create_retriever_tool(
        retriever_con_expansion,
        name="busqueda_libros_deep_learning",
        description="Busca y retorna información de los libros y papers locales de Machine Learning y Deep Learning. Úsalo SIEMPRE como primera opción para responder preguntas de la materia. Incluye el nombre del libro y la página en la respuesta.",
    )

    # 2. Herramienta ArXiv Ultrarrápida (Sin retries molestos)
    @tool
    def busqueda_arxiv_papers(query: str) -> str:
        """Busca papers académicos recientes en arXiv. Úsalo SÓLO si la información no se encontró en la herramienta de libros locales."""
        import urllib.parse
        import xml.etree.ElementTree as ET
        
        safe_query = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{safe_query}&start=0&max_results=3"
        
        try:
            # Timeout muy estricto de 3 segundos. Si ArXiv no responde, cortamos inmediatamente.
            response = requests.get(url, timeout=3)
            response.raise_for_status()
            
            root = ET.fromstring(response.text)
            results = []
            for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                title_el = entry.find("{http://www.w3.org/2005/Atom}title")
                summary_el = entry.find("{http://www.w3.org/2005/Atom}summary")
                
                title = title_el.text.replace("\n", " ").strip() if title_el is not None else "Sin título"
                summary = summary_el.text.replace("\n", " ").strip() if summary_el is not None else "Sin resumen"
                results.append(f"Título: {title}\nResumen: {summary}\n---")
                
            return "\n".join(results) if results else "No se encontraron resultados en ArXiv."
            
        except Exception as e:
            return "ERROR: ArXiv está temporalmente bloqueado o muy lento. Tienes estrictamente prohibido rendirte. DEBES usar OBLIGATORIAMENTE la herramienta 'busqueda_openalex_papers' AHORA MISMO con la misma consulta para buscar estos papers."

    # 4. Herramienta OpenAlex (Reemplazo antibloqueo para buscar papers)
    @tool
    def busqueda_openalex_papers(query: str) -> str:
        """Busca en la base de datos OpenAlex papers académicos. Úsalo para obtener resúmenes, métricas, conteo de citas o buscar literatura académica seria externa que no esté en tus libros."""
        url = "https://api.openalex.org/works"
        params = {"search": query, "per-page": 3}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data.get("results"):
                return "No se encontraron resultados en OpenAlex."
            
            results = []
            for paper in data["results"]:
                title = paper.get("title", "Sin título")
                year = paper.get("publication_year", "Año desconocido")
                cites = paper.get("cited_by_count", 0)
                
                # OpenAlex devuelve el abstract en formato de índice invertido para ahorrar ancho de banda
                abstract_idx = paper.get('abstract_inverted_index')
                abstract = "Sin resumen"
                if abstract_idx:
                    try:
                        max_pos = max([max(pos) for pos in abstract_idx.values()])
                        arr = [""] * (max_pos + 1)
                        for word, positions in abstract_idx.items():
                            for pos in positions:
                                arr[pos] = word
                        abstract = " ".join(arr)
                    except Exception:
                        pass
                
                authors_list = paper.get("authorships", [])
                authors = ", ".join([a["author"]["display_name"] for a in authors_list if "author" in a]) if authors_list else "Autores desconocidos"
                    
                results.append(f"Título: {title} ({year})\nAutores: {authors}\nCitas: {cites}\nResumen: {abstract[:800]}...\n---")
            return "\n".join(results)
        except Exception as e:
            return f"Error al consultar OpenAlex: {e}"

    tools = [busqueda_libros_deep_learning, busqueda_arxiv_papers, busqueda_openalex_papers]

    # 5. Configurar el LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )

    # 6. Definir el Prompt del Agente
    system_prompt = r"""Eres un asistente académico experto en Machine Learning y Deep Learning, diseñado para ayudar a estudiantes con sus dudas sobre la materia.

Instrucciones obligatorias:
1. LÍMITE DE TEMA (MUY ESTRICTO): Eres un tutor estrictamente académico. Si el usuario pregunta sobre algo que NO sea Machine Learning, Deep Learning, Inteligencia Artificial, programación o matemáticas (por ejemplo: recetas de cocina, deportes, etc.), DEBES negarte a responder inmediatamente. NO uses ninguna herramienta de búsqueda para estos temas.
2. JERARQUÍA DE BÚSQUEDA (ESTRICTA):
   - PRIMERO: Usa 'busqueda_libros_deep_learning' para CUALQUIER concepto, algoritmo o pregunta técnica de la materia.
   - SEGUNDO: Usa 'busqueda_arxiv_papers' o 'busqueda_openalex_papers' SOLO si te preguntan explícitamente por papers, investigaciones, métricas de citación, o literatura científica externa.
3. FORMATO MATEMÁTICO OBLIGATORIO: Si la respuesta requiere una fórmula matemática, DEBES formatearla estrictamente en LaTeX. Usa `$$` para ecuaciones en su propia línea (ejemplo: `$$ w = \sum \alpha_i x_i $$`) y `$` para variables dentro del texto (ejemplo: `$x_i$`). NUNCA uses texto plano como L_D = -E[log(D(x))].
4. Siempre debes CITAR de dónde sacaste la información indicando la herramienta que usaste y la fuente exacta.
5. Responde siempre en español de manera clara, didáctica y estructurada.
"""

    # 7. Crear el Agente (Sintaxis LangChain 1.3.1+)
    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    return graph
