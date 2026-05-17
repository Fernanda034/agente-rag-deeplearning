import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT_DIR / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_DB_DIR = ROOT_DIR / "vector_db"
COLLECTION_NAME = "deeplearning_corpus"


def build_filter(branch: str | None, topic: str | None, subtopic: str | None) -> dict[str, str] | None:
    filters = {}
    if branch:
        filters["branch"] = branch
    if topic:
        filters["topic"] = topic
    if subtopic:
        filters["subtopic"] = subtopic
    return filters or None


def get_vectorstore(db_dir: Path = DEFAULT_DB_DIR) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(db_dir),
        embedding_function=embeddings,
    )


def search(
    query: str,
    branch: str | None = "deep_learning",
    topic: str | None = None,
    subtopic: str | None = None,
    k: int = 4,
    db_dir: Path = DEFAULT_DB_DIR,
):
    vectorstore = get_vectorstore(db_dir)
    metadata_filter = build_filter(branch, topic, subtopic)
    kwargs = {"k": k}
    if metadata_filter:
        kwargs["filter"] = metadata_filter
    return vectorstore.similarity_search(query, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Busqueda semantica con filtros de metadatos.")
    parser.add_argument("query", help="Pregunta o concepto a buscar.")
    parser.add_argument("--branch", default="deep_learning")
    parser.add_argument("--topic")
    parser.add_argument("--subtopic")
    parser.add_argument("-k", type=int, default=4)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    results = search(
        args.query,
        branch=args.branch,
        topic=args.topic,
        subtopic=args.subtopic,
        k=args.k,
        db_dir=args.db_dir,
    )

    if not results:
        print("No se encontraron resultados.")
        return

    for index, doc in enumerate(results, 1):
        metadata = doc.metadata
        print(f"\n--- Resultado {index} ---")
        print(f"Rama: {metadata.get('branch')}")
        print(f"Tema: {metadata.get('topic')}")
        print(f"Subtema: {metadata.get('subtopic')}")
        print(f"Fuente: {metadata.get('file_name') or metadata.get('source')}")
        print(f"Pagina: {metadata.get('page')}")
        print(doc.page_content[:1200])


if __name__ == "__main__":
    main()
