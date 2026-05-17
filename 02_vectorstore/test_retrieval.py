import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from langchain_chroma import Chroma
from langchain.retrievers import MultiQueryRetriever
from langchain_groq import ChatGroq
from rag_utils.onnx_embeddings import ChromaONNXEmbeddings

load_dotenv()

DB_DIR = ROOT_DIR / "vector_db"


def test_retrieval():
    print("Cargando la base de datos vectorial para prueba...")

    if not DB_DIR.exists():
        print(f"Error: No se encontró la base de datos en {DB_DIR}. Debes correr ingestion.py primero.")
        return

    embeddings = ChromaONNXEmbeddings()

    vectorstore = Chroma(
        collection_name="deeplearning_corpus",
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )

    # MultiQueryRetriever: el LLM reescribe la pregunta en 3 variaciones para mejorar recall
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    retriever_avanzado = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        llm=llm,
    )

    print("\n¡Listo! Escribe 'salir' para terminar.")
    while True:
        query = input("\n🔎 Ingresa el concepto que quieres buscar: ")

        if query.lower() == "salir":
            break

        print(f"\nRealizando búsqueda AVANZADA (Llama 3 está reescribiendo la pregunta): '{query}'\n")

        results = retriever_avanzado.invoke(query)

        if not results:
            print("No se encontraron resultados.")
            continue

        print(f"Se encontraron {len(results)} fragmentos (chunks) relevantes:\n")

        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Desconocido")
            page = doc.metadata.get("page", "Desconocido")
            print(f"--- Fragmento {i} ---")
            print(f"Fuente: {source} (Página {page})")
            print(f"Texto extraído:\n{doc.page_content}\n")
            print("-" * 50)


if __name__ == "__main__":
    test_retrieval()
