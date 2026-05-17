import argparse
import csv
import os
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT_DIR / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.indexes._sql_record_manager import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.indexing import index
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CORPUS_DIR = ROOT_DIR / "corpus"
DEFAULT_MANIFEST = ROOT_DIR / "docs" / "metadatos_temas_corpus.csv"
DEFAULT_DB_DIR = ROOT_DIR / "vector_db"
COLLECTION_NAME = "deeplearning_corpus"


def clean_text(text: str) -> str:
    """Limpieza conservadora para texto extraido de PDFs academicos."""
    text = text.encode("utf-8", "ignore").decode("utf-8")
    replacements = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb00": "ff",
        "ï¬": "fi",
        "ï¬‚": "fl",
        "ï¬€": "ff",
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


def find_manifest_row(pdf_path: Path, manifest_rows: list[dict[str, str]]) -> dict[str, str]:
    pdf_name = pdf_path.name.lower()
    relative_path = normalize_path(str(pdf_path.relative_to(ROOT_DIR))) if pdf_path.is_relative_to(ROOT_DIR) else normalize_path(str(pdf_path))

    for row in manifest_rows:
        source = normalize_path(row.get("source_path_or_title", ""))
        if not source:
            continue
        if source.endswith(pdf_name) or source == relative_path:
            return row

    branch = "deep_learning" if "deep_learning" in relative_path else "machine_learning_estadistica"
    return {
        "id": pdf_path.stem,
        "branch": branch,
        "topic": "sin_clasificar",
        "subtopic": "sin_clasificar",
        "chapter_hint": "",
        "source_type": "pdf_local",
        "url": "",
        "scope": "sin_clasificar",
        "use_in_retrieval": "conditional",
        "priority": "media",
        "notes": "Fuente no encontrada en manifest; revisar metadatos.",
    }


def row_to_metadata(row: dict[str, str], pdf_path: Path) -> dict[str, str]:
    return {
        "manifest_id": row.get("id", pdf_path.stem),
        "branch": row.get("branch", "deep_learning"),
        "topic": row.get("topic", "sin_clasificar"),
        "subtopic": row.get("subtopic", "sin_clasificar"),
        "chapter": row.get("chapter_hint", ""),
        "source_type": row.get("source_type", "pdf_local"),
        "scope": row.get("scope", ""),
        "use_in_retrieval": row.get("use_in_retrieval", "conditional"),
        "priority": row.get("priority", ""),
        "source_url": row.get("url", ""),
        "source": str(pdf_path),
        "file_name": pdf_path.name,
    }


def load_pdf_documents(pdf_path: Path, base_metadata: dict[str, str]) -> list[Document]:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    cleaned_pages = []

    for page in pages:
        page.page_content = clean_text(page.page_content)
        if len(page.page_content) <= 50:
            continue

        metadata = dict(page.metadata)
        metadata.update(base_metadata)
        metadata["page"] = metadata.get("page", 0)
        cleaned_pages.append(Document(page_content=page.page_content, metadata=metadata))

    return cleaned_pages


def load_corpus(corpus_dir: Path, manifest_rows: list[dict[str, str]]) -> list[Document]:
    pdf_paths = sorted(corpus_dir.rglob("*.pdf"))
    documents = []

    for pdf_path in pdf_paths:
        row = find_manifest_row(pdf_path, manifest_rows)
        if row.get("use_in_retrieval") == "no":
            print(f"Omitiendo por metadatos: {pdf_path.name}")
            continue

        metadata = row_to_metadata(row, pdf_path)
        print(f"Cargando {pdf_path.name} -> {metadata['branch']} / {metadata['topic']}")
        documents.extend(load_pdf_documents(pdf_path, metadata))

    return documents


def write_ingestion_report(splits: list[Document], output_path: Path) -> None:
    counts: dict[tuple[str, str, str], int] = {}
    for doc in splits:
        key = (
            doc.metadata.get("branch", "sin_rama"),
            doc.metadata.get("topic", "sin_tema"),
            doc.metadata.get("subtopic", "sin_subtema"),
        )
        counts[key] = counts.get(key, 0) + 1

    lines = [
        "# Reporte de ingesta por metadatos",
        "",
        "| Rama | Tema | Subtema | Chunks |",
        "| --- | --- | --- | --- |",
    ]
    for (branch, topic, subtopic), count in sorted(counts.items()):
        lines.append(f"| `{branch}` | `{topic}` | `{subtopic}` | {count} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta RAG con metadatos academicos por tema.")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--report", type=Path, default=ROOT_DIR / "docs" / "reporte_ingesta_metadatos.md")
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")

    if not args.manifest.exists():
        raise FileNotFoundError(f"No existe el manifest: {args.manifest}")

    manifest_rows = load_manifest(args.manifest)
    documents = load_corpus(args.corpus_dir, manifest_rows)

    if not documents:
        print("No se encontraron paginas validas para indexar.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        length_function=len,
    )
    splits = [doc for doc in splitter.split_documents(documents) if len(doc.page_content) > 50]
    print(f"Chunks listos para vectorizar: {len(splits)}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    args.db_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(args.db_dir),
    )

    record_manager = SQLRecordManager(
        f"chroma/{COLLECTION_NAME}",
        db_url=f"sqlite:///{(args.db_dir / 'record_manager.sql').as_posix()}",
    )
    record_manager.create_schema()

    result = index(
        docs_source=splits,
        record_manager=record_manager,
        vector_store=vectorstore,
        cleanup="incremental",
        source_id_key="source",
    )

    write_ingestion_report(splits, args.report)
    print(f"Ingesta completada: {result}")
    print(f"Reporte escrito en: {args.report}")


if __name__ == "__main__":
    main()
