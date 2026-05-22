import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT_DIR / ".conda_packages_py39"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))
FALLBACK_PACKAGES = ROOT_DIR / ".python_packages"
if FALLBACK_PACKAGES.exists():
    sys.path.insert(0, str(FALLBACK_PACKAGES))

from ingestion_chroma_direct import collect_chunks, load_manifest
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings

def main():
    load_dotenv()
    
    # Verificar credenciales
    if not os.environ.get("PINECONE_API_KEY") or not os.environ.get("HF_TOKEN"):
        print("❌ Error: Faltan las credenciales. Asegúrate de configurar PINECONE_API_KEY y HF_TOKEN en el archivo .env.")
        sys.exit(1)
        
    print("1. Extrayendo chunks de los PDFs locales...")
    corpus_dir = ROOT_DIR / "corpus"
    manifest = ROOT_DIR / "docs" / "metadatos_temas_corpus.csv"
    
    manifest_rows = load_manifest(manifest)
    ids, texts, metadatas, counts = collect_chunks(corpus_dir, manifest_rows, 1000, 200)
    
    if not texts:
        print("No se encontraron textos para ingestar.")
        return
        
    print(f"Total chunks extraídos: {len(texts)}")
    
    print("2. Inicializando HuggingFaceEndpointEmbeddings (Remoto)...")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=os.environ.get("HF_TOKEN")
    )
    
    index_name = os.environ.get("PINECONE_INDEX_NAME", "deeplearning-corpus")
    
    print(f"3. Subiendo {len(texts)} chunks a Pinecone (Índice: {index_name}). Esto puede tardar...")
    
    # Subir a Pinecone en batches automáticamente usando LangChain
    PineconeVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        index_name=index_name,
        pinecone_api_key=os.environ.get("PINECONE_API_KEY")
    )
    
    print("✅ ¡Ingesta a Pinecone completada con éxito! Ya puedes eliminar la carpeta vector_db.")

if __name__ == "__main__":
    main()
