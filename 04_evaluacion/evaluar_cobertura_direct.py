import argparse
import csv
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT_DIR / ".python_packages"
sys.path.insert(0, str(ROOT_DIR))
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from sklearn.feature_extraction.text import HashingVectorizer

from rag_utils.query_expansion import expand_query_for_english_corpus


DEFAULT_MANIFEST = ROOT_DIR / "docs" / "metadatos_temas_corpus.csv"
DEFAULT_DB_DIR = ROOT_DIR / "vector_db"
DEFAULT_OUTPUT = ROOT_DIR / "docs" / "tabla_alcance_agente.md"
DEFAULT_ONNX_CACHE = ROOT_DIR / ".chroma_onnx_cache" / "onnx_models" / "all-MiniLM-L6-v2"
COLLECTION_NAME = "deeplearning_corpus"


TEST_QUESTIONS = {
    "entrenamiento_optimizacion": "Que problema resuelve Adam frente a descenso del gradiente clasico?",
    "regularizacion_estabilidad": "Para que sirven dropout y Batch Normalization en una red profunda?",
    "secuenciales_rnn_lstm_gru": "En que se diferencian RNN, LSTM y GRU?",
    "atencion_transformers": "Que es self-attention y para que sirve multi-head attention?",
    "cnn_vision": "Por que ResNet usa conexiones residuales en redes convolucionales profundas?",
    "explicabilidad": "Que muestra Grad-CAM en una red convolucional?",
    "generativos": "En que se diferencia un VAE de una GAN?",
}


def embed_query(query: str, n_features: int = 384) -> list[float]:
    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
    )
    return vectorizer.transform([query]).toarray().astype(float)[0].tolist()


def embed_query_onnx(query: str) -> list[float]:
    ONNXMiniLM_L6_V2.DOWNLOAD_PATH = str(DEFAULT_ONNX_CACHE)
    embedding_function = ONNXMiniLM_L6_V2()
    vector = embedding_function([query])[0]
    return vector.tolist() if hasattr(vector, "tolist") else list(vector)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def expected_topics(rows: list[dict[str, str]]) -> list[str]:
    return sorted(
        {
            row["topic"]
            for row in rows
            if row.get("branch") == "deep_learning" and row.get("use_in_retrieval") == "yes"
            and row.get("topic") != "general_deep_learning"
        }
    )


def level_from_results(documents: list[str], metadatas: list[dict]) -> tuple[int, str, str]:
    if not documents:
        return 0, "Hueco", "No recupera chunks utiles."
    technical_terms = [
        "activation",
        "attention",
        "batch",
        "convolution",
        "dropout",
        "embedding",
        "gate",
        "gradient",
        "latent",
        "loss",
        "normalization",
        "optimizer",
        "residual",
        "self-attention",
        "training",
        "transformer",
    ]
    scored = []
    for index, document in enumerate(documents):
        text = document.lower()
        term_count = sum(1 for term in technical_terms if term in text)
        has_depth = len(text) >= 300
        has_source = index < len(metadatas) and bool(metadatas[index].get("file_name"))
        score = term_count + (2 if has_depth else 0) + (1 if has_source else 0)
        scored.append((score, index, term_count, has_depth))

    best_score, best_index, best_terms, best_has_depth = max(scored, key=lambda item: item[0])
    source = metadatas[best_index].get("file_name", "sin fuente") if best_index < len(metadatas) else "sin fuente"
    useful_chunks = sum(1 for score, _, _, _ in scored if score >= 3)

    if best_terms >= 2 and best_has_depth:
        return 3, "Fuerte", f"Recupera contexto tecnico en top-{len(documents)} desde {source}."
    if best_score >= 3 or useful_chunks >= 2:
        return 2, "Basico", f"Recupera fragmentos utiles en top-{len(documents)} desde {source}."
    return 1, "Debil", f"Recupera texto relacionado, pero con poca profundidad desde {source}."


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabla de alcance usando ChromaDB directo.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--embedding-backend", choices=["onnx-minilm", "hashing"], default="onnx-minilm")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    topics = expected_topics(rows)
    collection = chromadb.PersistentClient(path=str(args.db_dir)).get_collection(COLLECTION_NAME)

    lines = [
        "# Tabla de alcance del agente",
        "",
        "| Tema | Pregunta de prueba | Nivel | Alcance | Observacion |",
        "| --- | --- | --- | --- | --- |",
    ]

    for topic in topics:
        question = TEST_QUESTIONS.get(topic, f"Explica el tema {topic} en Deep Learning.")
        question_for_embedding = expand_query_for_english_corpus(question)
        result = collection.query(
            query_embeddings=[embed_query_onnx(question_for_embedding) if args.embedding_backend == "onnx-minilm" else embed_query(question_for_embedding)],
            n_results=args.k,
            where={"$and": [{"branch": "deep_learning"}, {"topic": topic}]},
            include=["documents", "metadatas"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        level, label, observation = level_from_results(documents, metadatas)
        lines.append(f"| `{topic}` | {question} | {level} | {label} | {observation} |")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabla escrita en: {args.output}")


if __name__ == "__main__":
    main()
