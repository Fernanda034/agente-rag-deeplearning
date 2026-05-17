import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT_DIR / ".conda_packages_py39"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))
FALLBACK_PACKAGES = ROOT_DIR / ".python_packages"
if FALLBACK_PACKAGES.exists():
    sys.path.insert(0, str(FALLBACK_PACKAGES))

import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from pypdf import PdfReader
from sklearn.feature_extraction.text import HashingVectorizer


DEFAULT_CORPUS_DIR = ROOT_DIR / "corpus"
DEFAULT_MANIFEST = ROOT_DIR / "docs" / "metadatos_temas_corpus.csv"
DEFAULT_DB_DIR = ROOT_DIR / "vector_db"
DEFAULT_REPORT = ROOT_DIR / "docs" / "reporte_ingesta_chroma_direct.md"
DEFAULT_ONNX_CACHE = ROOT_DIR / ".chroma_onnx_cache" / "onnx_models" / "all-MiniLM-L6-v2"
COLLECTION_NAME = "deeplearning_corpus"


def clean_text(text: str) -> str:
    text = text.encode("utf-8", "ignore").decode("utf-8")
    replacements = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "ï¬€": "ff",
        "ï¬": "fi",
        "ï¬‚": "fl",
        "â€“": "-",
        "â€”": "-",
        "âˆ—": "*",
        "Â¨": "",
        "Â´": "'",
        "Â·": "-",
        "â€˜": "'",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def find_manifest_row(pdf_path: Path, rows: list[dict[str, str]]) -> dict[str, str]:
    rel = normalize_path(str(pdf_path.relative_to(ROOT_DIR)))
    name = pdf_path.name.lower()
    for row in rows:
        source = normalize_path(row.get("source_path_or_title", ""))
        if source == rel or source.endswith(name):
            return row

    return {
        "id": pdf_path.stem,
        "branch": "deep_learning" if "deep_learning" in rel else "machine_learning_estadistica",
        "topic": "sin_clasificar",
        "subtopic": "sin_clasificar",
        "chapter_hint": "",
        "source_type": "pdf_local",
        "url": "",
        "scope": "sin_clasificar",
        "use_in_retrieval": "conditional",
        "priority": "media",
        "notes": "Fuente no encontrada en manifest.",
    }


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks


def stable_id(source: str, page: int, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{page}:{chunk_index}:{text[:100]}".encode("utf-8")).hexdigest()
    return digest


def metadata_from_row(row: dict[str, str], pdf_path: Path, page: int) -> dict[str, str | int]:
    return {
        "manifest_id": row.get("id", pdf_path.stem) or pdf_path.stem,
        "branch": row.get("branch", "deep_learning") or "deep_learning",
        "topic": row.get("topic", "sin_clasificar") or "sin_clasificar",
        "subtopic": row.get("subtopic", "sin_clasificar") or "sin_clasificar",
        "chapter": row.get("chapter_hint", "") or "",
        "source_type": row.get("source_type", "pdf_local") or "pdf_local",
        "scope": row.get("scope", "") or "",
        "use_in_retrieval": row.get("use_in_retrieval", "conditional") or "conditional",
        "priority": row.get("priority", "") or "",
        "source_url": row.get("url", "") or "",
        "source": str(pdf_path.relative_to(ROOT_DIR)),
        "file_name": pdf_path.name,
        "page": page,
    }


def collect_chunks(corpus_dir: Path, manifest_rows: list[dict[str, str]], chunk_size: int, overlap: int):
    ids = []
    texts = []
    metadatas = []
    counts: dict[tuple[str, str, str], int] = {}

    for pdf_path in sorted(corpus_dir.rglob("*.pdf")):
        row = find_manifest_row(pdf_path, manifest_rows)
        if row.get("use_in_retrieval") == "no":
            print(f"Omitiendo por metadata: {pdf_path.name}")
            continue

        print(f"Procesando {pdf_path.name} -> {row.get('branch')} / {row.get('topic')}")
        reader = PdfReader(str(pdf_path))
        for page_index, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if len(text) <= 50:
                continue
            for chunk_index, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
                metadata = metadata_from_row(row, pdf_path, page_index)
                ids.append(stable_id(metadata["source"], page_index, chunk_index, chunk))
                texts.append(chunk)
                metadatas.append(metadata)
                key = (metadata["branch"], metadata["topic"], metadata["subtopic"])
                counts[key] = counts.get(key, 0) + 1

    return ids, texts, metadatas, counts


def write_report(counts: dict[tuple[str, str, str], int], output: Path, embedding_backend: str) -> None:
    lines = [
        "# Reporte de ingesta directa en ChromaDB",
        "",
        f"- Embeddings usados: `{embedding_backend}`",
        "- Nota: `onnx-minilm` usa embeddings semanticos locales con `all-MiniLM-L6-v2`; `hashing` queda solo como respaldo tecnico.",
        "",
        "| Rama | Tema | Subtema | Chunks |",
        "| --- | --- | --- | --- |",
    ]
    for (branch, topic, subtopic), count in sorted(counts.items()):
        lines.append(f"| `{branch}` | `{topic}` | `{subtopic}` | {count} |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def embed_texts_hashing(texts: list[str], n_features: int = 384) -> list[list[float]]:
    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        stop_words=None,
    )
    matrix = vectorizer.transform(texts)
    return matrix.toarray().astype(float).tolist()


def embed_texts_sentence_transformers(texts: list[str], model_name: str, batch_size: int) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True).tolist()


def embed_texts_onnx_minilm(texts: list[str], batch_size: int) -> list[list[float]]:
    ONNXMiniLM_L6_V2.DOWNLOAD_PATH = str(DEFAULT_ONNX_CACHE)
    embedding_function = ONNXMiniLM_L6_V2()
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors = embedding_function(batch)
        embeddings.extend([vector.tolist() if hasattr(vector, "tolist") else list(vector) for vector in vectors])
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta directa a ChromaDB con metadatos academicos.")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument(
        "--embedding-backend",
        choices=["onnx-minilm", "hashing", "sentence-transformers"],
        default="onnx-minilm",
    )
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    manifest_rows = load_manifest(args.manifest)
    ids, texts, metadatas, counts = collect_chunks(
        args.corpus_dir,
        manifest_rows,
        args.chunk_size,
        args.chunk_overlap,
    )

    if not texts:
        print("No se encontraron chunks para indexar.")
        return

    print(f"Total chunks: {len(texts)}")
    if args.embedding_backend == "onnx-minilm":
        print("Creando embeddings semanticos con Chroma ONNX MiniLM all-MiniLM-L6-v2.")
        embeddings = embed_texts_onnx_minilm(texts, args.batch_size)
    elif args.embedding_backend == "sentence-transformers":
        print(f"Cargando modelo embeddings: {args.model}")
        embeddings = embed_texts_sentence_transformers(texts, args.model, args.batch_size)
    else:
        print("Creando embeddings locales con HashingVectorizer.")
        embeddings = embed_texts_hashing(texts)

    client = chromadb.PersistentClient(path=str(args.db_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    for start in range(0, len(texts), args.batch_size):
        end = start + args.batch_size
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

    write_report(counts, args.report, args.embedding_backend)
    print(f"Ingesta completada en: {args.db_dir}")
    print(f"Reporte escrito en: {args.report}")


if __name__ == "__main__":
    main()
