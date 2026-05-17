import argparse
import csv
import re
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
DEFAULT_OUTPUT = ROOT_DIR / "docs" / "evaluacion_profundidad_chunks.md"
DEFAULT_ONNX_CACHE = ROOT_DIR / ".chroma_onnx_cache" / "onnx_models" / "all-MiniLM-L6-v2"
COLLECTION_NAME = "deeplearning_corpus"


TEST_SUITE = {
    "atencion_transformers": [
        ("definicion", "Que problema resuelve self-attention frente a modelos recurrentes?"),
        ("comparacion", "Compara self-attention, multi-head attention y positional encoding."),
        ("aplicacion", "Cuando conviene usar un Transformer encoder y cuando un decoder?"),
        ("diagnostico", "Que limitaciones computacionales tiene attention en secuencias largas?"),
    ],
    "cnn_vision": [
        ("definicion", "Que es una conexion residual en ResNet?"),
        ("comparacion", "Compara una red CNN profunda convencional con una ResNet."),
        ("aplicacion", "Cuando conviene usar ResNet o transfer learning en vision por computador?"),
        ("diagnostico", "Por que una red profunda puede degradar su entrenamiento y como ayuda ResNet?"),
    ],
    "entrenamiento_optimizacion": [
        ("definicion", "Que es Adam y que estimaciones mantiene durante el entrenamiento?"),
        ("comparacion", "Compara Adam con SGD, Momentum y RMSProp."),
        ("aplicacion", "Cuando conviene usar Adam en lugar de descenso del gradiente clasico?"),
        ("diagnostico", "Que problemas de tasa de aprendizaje o gradientes intenta mitigar Adam?"),
    ],
    "explicabilidad": [
        ("definicion", "Que es Grad-CAM y que mapa produce?"),
        ("comparacion", "Compara Grad-CAM con saliency maps o class activation maps."),
        ("aplicacion", "Como se usa Grad-CAM para interpretar una prediccion de una CNN?"),
        ("diagnostico", "Que limitaciones tiene Grad-CAM para explicar decisiones del modelo?"),
    ],
    "generativos": [
        ("definicion", "Que es una variable latente en un VAE y que aprende el modelo?"),
        ("comparacion", "Compara VAE y GAN en objetivo de entrenamiento y tipo de salida."),
        ("aplicacion", "Cuando conviene usar un VAE y cuando una GAN?"),
        ("diagnostico", "Que problemas de estabilidad o evaluacion aparecen en modelos generativos?"),
    ],
    "regularizacion_estabilidad": [
        ("definicion", "Que es dropout y como reduce el sobreajuste?"),
        ("comparacion", "Compara dropout, Batch Normalization y weight decay."),
        ("aplicacion", "Cuando conviene usar BatchNorm durante el entrenamiento de redes profundas?"),
        ("diagnostico", "Como se relacionan normalizacion, gradientes inestables y velocidad de entrenamiento?"),
    ],
    "secuenciales_rnn_lstm_gru": [
        ("definicion", "Que es una GRU y que puertas utiliza?"),
        ("comparacion", "Compara RNN simple, LSTM y GRU."),
        ("aplicacion", "Cuando conviene usar un modelo recurrente para secuencias o traduccion?"),
        ("diagnostico", "Como ayudan las compuertas a mitigar vanishing gradients en secuencias largas?"),
    ],
}


DEEP_TERMS = {
    "atencion_transformers": [
        "self-attention",
        "multi-head",
        "positional",
        "encoder",
        "decoder",
        "complexity",
        "sequence",
        "representation",
    ],
    "cnn_vision": [
        "residual",
        "degradation",
        "identity",
        "convolution",
        "deep",
        "training error",
        "shortcut",
        "layers",
    ],
    "entrenamiento_optimizacion": [
        "adam",
        "gradient",
        "moment",
        "learning rate",
        "stochastic",
        "adaptive",
        "rmsprop",
        "momentum",
    ],
    "explicabilidad": [
        "grad-cam",
        "gradient",
        "localization",
        "activation",
        "class-discriminative",
        "visual explanation",
        "convolutional",
    ],
    "generativos": [
        "variational",
        "latent",
        "generator",
        "discriminator",
        "adversarial",
        "inference",
        "lower bound",
        "auto-encoding",
    ],
    "regularizacion_estabilidad": [
        "dropout",
        "normalization",
        "overfitting",
        "covariate shift",
        "regularization",
        "mini-batch",
        "gradient",
        "training",
    ],
    "secuenciales_rnn_lstm_gru": [
        "gru",
        "gate",
        "reset",
        "update",
        "recurrent",
        "sequence",
        "encoder-decoder",
        "hidden state",
    ],
}


QUALITY_PATTERNS = {
    "formula_or_objective": re.compile(r"[=\u2211\u2207]|\bobjective\b|\bloss\b|\bbound\b|\bprobability\b", re.I),
    "comparison": re.compile(r"\bcompared?\b|\bversus\b|\bwhereas\b|\bunlike\b|\bdifferent\b|\bsimilar\b", re.I),
    "limitation": re.compile(r"\blimitation\b|\bhowever\b|\balthough\b|\bproblem\b|\bdifficult\b|\bchallenge\b|\bunstable\b", re.I),
    "application": re.compile(r"\buse\b|\btask\b|\bapplication\b|\bexperiment\b|\bimage\b|\btranslation\b|\bclassification\b", re.I),
}


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def expected_topics(rows: list[dict[str, str]]) -> list[str]:
    topics = {
        row["topic"]
        for row in rows
        if row.get("branch") == "deep_learning"
        and row.get("use_in_retrieval") == "yes"
        and row.get("topic") != "general_deep_learning"
    }
    return sorted(topic for topic in topics if topic in TEST_SUITE)


def embed_query_hashing(query: str, n_features: int = 384) -> list[float]:
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


def build_embedding(query: str, backend: str) -> list[float]:
    if backend == "onnx-minilm":
        return embed_query_onnx(query)
    return embed_query_hashing(query)


def score_documents(topic: str, question_type: str, documents: list[str], metadatas: list[dict]) -> tuple[int, str]:
    if not documents:
        return 0, "No recupera chunks utiles."

    text = "\n".join(documents[:3]).lower()
    terms = DEEP_TERMS.get(topic, [])
    term_hits = sum(1 for term in terms if term in text)
    source_count = sum(1 for metadata in metadatas[:3] if metadata.get("file_name") and metadata.get("page"))
    avg_len = sum(len(doc) for doc in documents[:3]) / max(1, min(3, len(documents)))
    pattern_hits = sum(1 for pattern in QUALITY_PATTERNS.values() if pattern.search(text))

    type_bonus = 0
    if question_type == "comparacion" and QUALITY_PATTERNS["comparison"].search(text):
        type_bonus += 1
    if question_type == "aplicacion" and QUALITY_PATTERNS["application"].search(text):
        type_bonus += 1
    if question_type == "diagnostico" and QUALITY_PATTERNS["limitation"].search(text):
        type_bonus += 1
    if QUALITY_PATTERNS["formula_or_objective"].search(text):
        type_bonus += 1

    raw = term_hits + pattern_hits + type_bonus
    has_context = avg_len >= 500
    has_sources = source_count >= 2

    if raw >= 8 and has_context and has_sources:
        level = 4
    elif raw >= 6 and has_context and has_sources:
        level = 3
    elif raw >= 4 and source_count >= 1:
        level = 2
    elif raw >= 2:
        level = 1
    else:
        level = 0

    best_source = "sin fuente"
    if metadatas:
        first = metadatas[0]
        best_source = f"{first.get('file_name', 'sin fuente')} p.{first.get('page', '?')}"

    detail = (
        f"terminos={term_hits}, senales_profundidad={pattern_hits}, "
        f"fuentes_top3={source_count}, fuente_top1={best_source}"
    )
    return level, detail


def label_from_average(value: float) -> str:
    if value >= 3.5:
        return "Profundo"
    if value >= 2.5:
        return "Fuerte"
    if value >= 1.5:
        return "Basico"
    if value > 0:
        return "Debil"
    return "Hueco"


def limitation_for(topic: str, min_level: int, avg_level: float) -> str:
    limitations = {
        "cnn_vision": "Fuerte en ResNet; CNN basica y transfer learning didactico pueden requerir fuente adicional.",
        "atencion_transformers": "Puede requerir refuerzo para complejidad computacional o variantes modernas.",
        "entrenamiento_optimizacion": "Fuerte en Adam; comparacion completa con SGD/Momentum/RMSProp puede requerir capitulo didactico.",
        "explicabilidad": "Fuerte en Grad-CAM; saliency maps generales y limites filosoficos quedan menos cubiertos.",
        "generativos": "Cubre VAE/GAN; autoencoders basicos, denoising y estabilidad practica pueden requerir complemento.",
        "regularizacion_estabilidad": "Fuerte en Dropout/BatchNorm; weight decay, early stopping e inicializacion quedan menos cubiertos.",
        "secuenciales_rnn_lstm_gru": "Cubre GRU; LSTM, RNN simple y vanishing gradients necesitan fuente local mas directa.",
    }
    if topic in limitations:
        return limitations[topic]
    if min_level >= 3 and avg_level >= 3.5:
        return "Apto para preguntas exigentes de demo."
    return limitations.get(topic, "Revisar con mas preguntas especificas.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluacion profunda de chunks RAG por tema.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embedding-backend", choices=["onnx-minilm", "hashing"], default="onnx-minilm")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    topics = expected_topics(load_manifest(args.manifest))
    collection = chromadb.PersistentClient(path=str(args.db_dir)).get_collection(COLLECTION_NAME)

    rows = []
    summaries = []

    for topic in topics:
        topic_levels = []
        topic_sources = set()
        for question_type, question in TEST_SUITE[topic]:
            expanded = expand_query_for_english_corpus(question)
            result = collection.query(
                query_embeddings=[build_embedding(expanded, args.embedding_backend)],
                n_results=args.k,
                where={"$and": [{"branch": "deep_learning"}, {"topic": topic}]},
                include=["documents", "metadatas"],
            )
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            level, detail = score_documents(topic, question_type, documents, metadatas)
            topic_levels.append(level)
            for metadata in metadatas[:3]:
                if metadata.get("file_name"):
                    topic_sources.add(metadata["file_name"])
            rows.append((topic, question_type, question, level, detail))

        avg_level = sum(topic_levels) / len(topic_levels)
        min_level = min(topic_levels)
        summaries.append(
            (
                topic,
                ", ".join(sorted(topic_sources)) or "sin fuente",
                avg_level,
                min_level,
                label_from_average(avg_level),
                limitation_for(topic, min_level, avg_level),
            )
        )

    lines = [
        "# Evaluacion profunda de chunks RAG",
        "",
        f"- Backend de embeddings: `{args.embedding_backend}`",
        f"- Top-k por pregunta: `{args.k}`",
        "- Escala: 0 hueco, 1 superficial, 2 clase basica, 3 tecnico, 4 profundo/comparativo.",
        "",
        "## Resumen por tema",
        "",
        "| Tema | Fuentes recuperadas | Promedio | Minimo | Alcance | Limitacion |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for topic, sources, avg_level, min_level, label, limitation in summaries:
        lines.append(f"| `{topic}` | {sources} | {avg_level:.2f} | {min_level} | {label} | {limitation} |")

    lines.extend(
        [
            "",
            "## Preguntas exigentes",
            "",
            "| Tema | Tipo | Pregunta | Nivel | Evidencia |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for topic, question_type, question, level, detail in rows:
        lines.append(f"| `{topic}` | {question_type} | {question} | {level} | {detail} |")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Evaluacion escrita en: {args.output}")


if __name__ == "__main__":
    main()
