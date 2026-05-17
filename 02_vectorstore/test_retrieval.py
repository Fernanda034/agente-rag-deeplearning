import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "../vector_db"

def test_retrieval():
    print("Cargando la base de datos vectorial para prueba...")
    
    if not os.path.exists(DB_DIR):
        print(f"Error: No se encontró la base de datos en {DB_DIR}. Debes correr ingestion.py primero.")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma(
        collection_name="deeplearning_corpus",
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    
    # Configurar el LLM y el MultiQueryRetriever (Expansión de pregunta)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    retriever_avanzado = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        llm=llm
    )
    
    print("\n¡Listo! Escribe 'salir' para terminar.")
    while True:
        # Hacer una búsqueda de prueba interactiva
        query = input("\n🔎 Ingresa el concepto que quieres buscar: ")
        
        if query.lower() == 'salir':
            break
            
        print(f"\nRealizando búsqueda AVANZADA (Llama 3 está reescribiendo la pregunta): '{query}'\n")
        
        # Realizar búsqueda con expansión de pregunta
        results = retriever_avanzado.invoke(query)
        
        if not results:
            print("No se encontraron resultados.")
            continue
            
        print(f"Se encontraron {len(results)} fragmentos (chunks) relevantes:\n")
        
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'Desconocido')
            page = doc.metadata.get('page', 'Desconocido')
            print(f"--- Fragmento {i} ---")
            print(f"Fuente: {source} (Página {page})")
            print(f"Texto extraído:\n{doc.page_content}\n")
            print("-" * 50)

if __name__ == "__main__":
    test_retrieval()
