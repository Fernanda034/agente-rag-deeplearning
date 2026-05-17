import argparse
import csv
import re
from pathlib import Path

from pypdf import PdfReader


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = ROOT_DIR / "corpus"
DEFAULT_MANIFEST = ROOT_DIR / "docs" / "metadatos_temas_corpus.csv"
DEFAULT_REPORT = ROOT_DIR / "docs" / "reporte_preprocesamiento_preview.md"


def clean_text(text: str) -> str:
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
        "use_in_retrieval": "conditional",
    }


def chunk_count(text: str, chunk_size: int, chunk_overlap: int) -> int:
    if not text:
        return 0
    step = max(1, chunk_size - chunk_overlap)
    return max(1, (max(0, len(text) - chunk_overlap) + step - 1) // step)


def preview_pdf(pdf_path: Path, row: dict[str, str], chunk_size: int, chunk_overlap: int) -> dict[str, object]:
    reader = PdfReader(str(pdf_path))
    valid_pages = 0
    chars = 0
    chunks = 0
    sample = ""

    for page in reader.pages:
        text = clean_text(page.extract_text() or "")
        if len(text) <= 50:
            continue
        valid_pages += 1
        chars += len(text)
        chunks += chunk_count(text, chunk_size, chunk_overlap)
        if not sample:
            sample = text[:300]

    return {
        "file": str(pdf_path.relative_to(ROOT_DIR)),
        "branch": row.get("branch", ""),
        "topic": row.get("topic", ""),
        "subtopic": row.get("subtopic", ""),
        "pages": valid_pages,
        "chars": chars,
        "chunks": chunks,
        "sample": sample.replace("|", "\\|"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview ligero de preprocesamiento y metadatos.")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    pdfs = sorted(args.corpus_dir.rglob("*.pdf"))
    previews = []

    for pdf in pdfs:
        row = find_manifest_row(pdf, rows)
        if row.get("use_in_retrieval") == "no":
            continue
        previews.append(preview_pdf(pdf, row, args.chunk_size, args.chunk_overlap))

    lines = [
        "# Preview de preprocesamiento con metadatos",
        "",
        f"Chunk size: `{args.chunk_size}`",
        f"Chunk overlap: `{args.chunk_overlap}`",
        "",
        "| Archivo | Rama | Tema | Subtema | Paginas validas | Chunks estimados |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for item in previews:
        lines.append(
            f"| `{item['file']}` | `{item['branch']}` | `{item['topic']}` | "
            f"`{item['subtopic']}` | {item['pages']} | {item['chunks']} |"
        )

    lines.extend(["", "## Muestras de texto limpio", ""])
    for item in previews:
        lines.append(f"### `{item['file']}`")
        lines.append("")
        lines.append(item["sample"] or "Sin muestra disponible.")
        lines.append("")

    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Preview escrito en: {args.report}")


if __name__ == "__main__":
    main()
