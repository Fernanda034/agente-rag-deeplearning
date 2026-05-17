import argparse
import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT_DIR / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_MANIFEST = ROOT_DIR / "docs" / "metadatos_temas_corpus.csv"
DEFAULT_DB_DIR = ROOT_DIR / "vector_db"
DEFAULT_OUTPUT = ROOT_DIR / "docs" / "tabla_alcance_agente.md"
COLLECTION_NAME = "deeplearning_corpus"


TEST_QUESTIONS = {
    "fundamentos_mlp": "Que es un perceptron multicapa y para que sirven las funciones de activacion?",
    "entrenamiento_optimizacion": "Que problema resuelve Adam frente a descenso del gradiente clasico?",
    "regularizacion_estabilidad": "Para que sirven dropout y Batch Normalization en una red profunda?",
    "secuenciales_rnn_lstm_gru": "En que se diferencian RNN, LSTM y GRU?",
    "atencion_transformers": "Que es self-attention y para que sirve multi-head attention?",
    "cnn_vision": "Por que ResNet usa conexiones residuales en redes convolucionales profundas?",
    "explicabilidad": "Que muestra Grad-CAM en una red convolucional?",
    "generativos": "En que se diferencia un VAE de una GAN?",
}


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def expected_topics(rows: list[dict[str, str]]) -> list[str]:
    topics = {
        row["topic"]
        for row in rows
        if row.get("branch") == "deep_learning" and row.get("use_in_retrieval") == "yes"
    }
    return sorted(topics)


def depth_level(results) -> tuple[int, str]:
    if not results:
        return 0, "No recupera chunks utiles."

    best_text = results[0].page_content
    if len(best_text) < 300:
        return 1, "Recupera fragmentos cortos o superficiales."
    if any(token in best_text.lower() for token in ["formula", "loss", "gradient", "gate", "attention", "convolution", "latent"]):
        return 3, "Recupera fragmentos tecnicos con buena profundidad."
    return 2, "Recupera explicacion con contexto."


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera tabla de alcance del agente por tema.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    topics = expected_topics(rows)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(args.db_dir),
        embedding_function=embeddings,
    )

    lines = [
        "# Tabla de alcance del agente",
        "",
        "| Tema | Pregunta de prueba | Nivel | Puede responder | Limitacion |",
        "| --- | --- | --- | --- | --- |",
    ]

    for topic in topics:
        question = TEST_QUESTIONS.get(topic, f"Explica el tema {topic} en Deep Learning.")
        results = vectorstore.similarity_search(
            question,
            k=args.k,
            filter={"branch": "deep_learning", "topic": topic},
        )
        level, note = depth_level(results)
        can_answer = "Fuerte" if level >= 3 else "Basico" if level == 2 else "Debil" if level == 1 else "Hueco"
        limitation = note if level < 3 else "Validar con preguntas mas exigentes antes de demo."
        lines.append(f"| `{topic}` | {question} | {level} | {can_answer} | {limitation} |")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabla escrita en: {args.output}")


if __name__ == "__main__":
    main()
