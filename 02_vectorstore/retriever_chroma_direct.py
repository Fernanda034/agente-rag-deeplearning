import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT_DIR / ".python_packages"
sys.path.insert(0, str(ROOT_DIR))
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from sklearn.feature_extraction.text import HashingVectorizer

from rag_utils.query_expansion import expand_query_for_english_corpus


DEFAULT_DB_DIR = ROOT_DIR / "vector_db"
DEFAULT_ONNX_CACHE = ROOT_DIR / ".chroma_onnx_cache" / "onnx_models" / "all-MiniLM-L6-v2"
COLLECTION_NAME = "deeplearning_corpus"


def embed_query(query: str, n_features: int = 384) -> list[float]:
    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        stop_words=None,
    )
    return vectorizer.transform([query]).toarray().astype(float)[0].tolist()


def embed_query_onnx(query: str) -> list[float]:
    ONNXMiniLM_L6_V2.DOWNLOAD_PATH = str(DEFAULT_ONNX_CACHE)
    embedding_function = ONNXMiniLM_L6_V2()
    vector = embedding_function([query])[0]
    return vector.tolist() if hasattr(vector, "tolist") else list(vector)


def build_where(branch: str | None, topic: str | None, subtopic: str | None):
    clauses = []
    if branch:
        clauses.append({"branch": branch})
    if topic:
        clauses.append({"topic": topic})
    if subtopic:
        clauses.append({"subtopic": subtopic})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def main() -> None:
    parser = argparse.ArgumentParser(description="Busqueda directa en ChromaDB con filtros de metadatos.")
    parser.add_argument("query")
    parser.add_argument("--branch", default="deep_learning")
    parser.add_argument("--topic")
    parser.add_argument("--subtopic")
    parser.add_argument("-k", type=int, default=4)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--embedding-backend", choices=["onnx-minilm", "hashing"], default="onnx-minilm")
    parser.add_argument("--no-query-expansion", action="store_true")
    args = parser.parse_args()

    query_for_embedding = args.query if args.no_query_expansion else expand_query_for_english_corpus(args.query)
    client = chromadb.PersistentClient(path=str(args.db_dir))
    collection = client.get_collection(COLLECTION_NAME)
    result = collection.query(
        query_embeddings=[embed_query_onnx(query_for_embedding) if args.embedding_backend == "onnx-minilm" else embed_query(query_for_embedding)],
        n_results=args.k,
        where=build_where(args.branch, args.topic, args.subtopic),
        include=["documents", "metadatas", "distances"],
    )

    docs = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    if not docs:
        print("No se encontraron resultados.")
        return

    if query_for_embedding != args.query:
        print("Consulta enriquecida para corpus en ingles:")
        print(query_for_embedding)

    for index, (doc, metadata, distance) in enumerate(zip(docs, metadatas, distances), 1):
        print(f"\n--- Resultado {index} ---")
        print(f"Distancia: {distance:.4f}")
        print(f"Rama: {metadata.get('branch')}")
        print(f"Tema: {metadata.get('topic')}")
        print(f"Subtema: {metadata.get('subtopic')}")
        print(f"Fuente: {metadata.get('file_name')} pagina {metadata.get('page')}")
        safe_doc = doc[:1200].encode("cp1252", errors="replace").decode("cp1252")
        print(safe_doc)


if __name__ == "__main__":
    main()
