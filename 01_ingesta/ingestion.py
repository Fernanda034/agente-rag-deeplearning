import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.indexes._sql_record_manager import SQLRecordManager
from langchain_core.indexing import index

# Cargar variables de entorno
load_dotenv()

CORPUS_DIR = "../corpus"
DB_DIR = "../vector_db"
RECORD_MANAGER_DB_URL = "sqlite:///../vector_db/record_manager.sql"

def clean_text(text: str) -> str:
    """Aplica expresiones regulares para limpiar el ruido del texto de los PDFs."""
    # 0. Eliminar caracteres unicode subrogados (surrogates) mal formados por PyPDF
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    # 1. Unir palabras separadas por guion al final del renglón (ej. "hiper- \n parámetro" -> "hiperparámetro")
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # 2. Quitar saltos de línea sencillos dentro de párrafos para no cortar oraciones.
    # Deja intactos los saltos de línea dobles que sí separan párrafos reales.
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # 3. Limpiar espacios múltiples y tabulaciones
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    print("Iniciando proceso de ingesta AVANZADA de documentos...")
    
    print(f"Buscando PDFs en la carpeta: {CORPUS_DIR}")
    loader = PyPDFDirectoryLoader(CORPUS_DIR)
    docs = loader.load()
    
    if not docs:
        print("No se encontraron documentos en la carpeta corpus. Revisa la ruta.")
        return
        
    print(f"Se cargaron {len(docs)} páginas en total. Iniciando limpieza exhaustiva...")
    
    # Aplicar limpieza y filtrar páginas basura (ej. en blanco o solo con un número)
    valid_docs = []
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        if len(doc.page_content) > 50:
            valid_docs.append(doc)
            
    print("Dividiendo los documentos limpios en fragmentos (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(valid_docs)
    
    # Filtro final de chunks basura
    final_splits = [s for s in splits if len(s.page_content) > 50]
    print(f"Se obtuvieron {len(final_splits)} fragmentos puros listos para vectorizar.")
    
    print("Cargando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 1. Crear/Cargar el Vector Store
    print(f"Conectando con la base de datos vectorial en: {DB_DIR}")
    os.makedirs(DB_DIR, exist_ok=True)
    vectorstore = Chroma(
        collection_name="deeplearning_corpus",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    
    # 2. Inicializar el Record Manager (SQLite)
    namespace = "chroma/deeplearning_corpus"
    record_manager = SQLRecordManager(namespace, db_url=RECORD_MANAGER_DB_URL)
    record_manager.create_schema()
    
    # 3. Indexación Incremental (Magia Anti-duplicados)
    print("Sincronizando documentos en la base de datos (Indexación Incremental)...")
    print("Esto puede tardar varios minutos dependiendo del peso de los PDFs.")
    
    result = index(
        docs_source=final_splits,
        record_manager=record_manager,
        vector_store=vectorstore,
        cleanup="incremental",
        source_id_key="source" # Clave en metadata que identifica de qué archivo viene
    )
    
    print("\n¡Ingesta completada con éxito!")
    print(f"Resumen de la sincronización: {result}")
    # El result te dirá {'num_added': X, 'num_updated': Y, 'num_skipped': Z, 'num_deleted': W}

if __name__ == "__main__":
    main()
