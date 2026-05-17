"""
Evaluación RAGAS del pipeline RAG completo.

Usa Groq (Llama 3.3 70b) como juez LLM y all-MiniLM-L6-v2 como modelo
de embeddings para calcular las cuatro métricas núcleo de RAGAS 0.4.x:
  - faithfulness        : la respuesta es fiel al contexto recuperado
  - answer_relevancy    : la respuesta es relevante a la pregunta
  - context_precision   : los chunks recuperados son pertinentes
  - context_recall      : el contexto cubre la respuesta esperada

Uso:
    python 04_evaluacion/ragas_eval.py
    python 04_evaluacion/ragas_eval.py --output docs/ragas_report.md --k 4
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from rag_utils.query_expansion import expand_query_for_english_corpus


DEFAULT_DB_DIR = ROOT_DIR / "vector_db"
DEFAULT_ONNX_CACHE = ROOT_DIR / ".chroma_onnx_cache" / "onnx_models" / "all-MiniLM-L6-v2"
DEFAULT_OUTPUT = ROOT_DIR / "docs" / "ragas_report.md"
COLLECTION_NAME = "deeplearning_corpus"

# Preguntas de evaluación con respuesta de referencia (ground truth)
EVAL_SUITE = [
    {
        "topic": "atencion_transformers",
        "question": "¿Qué problema resuelve self-attention frente a los modelos recurrentes y cómo funciona multi-head attention?",
        "ground_truth": (
            "Self-attention permite que cada token de la secuencia atienda a todos los demás tokens en paralelo, "
            "eliminando la dependencia secuencial de las RNNs y resolviendo el problema del gradiente que desaparece "
            "en secuencias largas. Multi-head attention aplica varias cabezas de atención en paralelo con distintas "
            "proyecciones lineales, capturando relaciones semánticas desde múltiples subespacios de representación."
        ),
    },
    {
        "topic": "entrenamiento_optimizacion",
        "question": "¿Qué estimaciones mantiene Adam durante el entrenamiento y en qué se diferencia de SGD con Momentum?",
        "ground_truth": (
            "Adam mantiene dos estimaciones de momentos: el primer momento (media móvil de los gradientes, similar a "
            "Momentum) y el segundo momento (media móvil de los gradientes al cuadrado, similar a RMSProp). Ambas se "
            "corrigen por el sesgo de inicialización. La diferencia con SGD+Momentum es que Adam adapta la tasa de "
            "aprendizaje individualmente para cada parámetro basándose en la magnitud histórica del gradiente."
        ),
    },
    {
        "topic": "regularizacion_estabilidad",
        "question": "¿Cómo reduce Dropout el sobreajuste y en qué se diferencia de Batch Normalization?",
        "ground_truth": (
            "Dropout desactiva aleatoriamente neuronas durante el entrenamiento con probabilidad p, forzando al modelo "
            "a aprender representaciones redundantes y robustas, lo que actúa como regularización implícita. "
            "Batch Normalization, en cambio, normaliza las activaciones de cada mini-batch para reducir el cambio "
            "covariable interno, estabilizando el entrenamiento y permitiendo tasas de aprendizaje más altas. "
            "Dropout actúa sobre las neuronas; BatchNorm actúa sobre la distribución de las activaciones."
        ),
    },
    {
        "topic": "cnn_vision",
        "question": "¿Por qué las redes profundas convencionales sufren degradación del rendimiento y cómo lo resuelve ResNet?",
        "ground_truth": (
            "En redes muy profundas sin conexiones residuales, el error de entrenamiento puede aumentar al agregar más "
            "capas debido al problema de la degradación (no confundir con sobreajuste). ResNet introduce conexiones "
            "de atajo (skip connections) que suman la entrada de un bloque a su salida, permitiendo que la red aprenda "
            "residuos F(x) en lugar de mapeos directos H(x). Esto facilita el flujo del gradiente y permite entrenar "
            "redes de más de 100 capas exitosamente."
        ),
    },
    {
        "topic": "generativos",
        "question": "¿En qué se diferencian un VAE y una GAN en su objetivo de entrenamiento y tipo de salida generada?",
        "ground_truth": (
            "Un VAE (Variational Autoencoder) maximiza un límite inferior de la log-verosimilitud de los datos (ELBO) "
            "mediante inferencia variacional, aprendiendo una distribución latente explícita y generando muestras suaves "
            "e interpolables. Una GAN (Generative Adversarial Network) entrena un generador y un discriminador en un "
            "juego adversarial minimax, produciendo muestras nítidas pero con riesgo de colapso de modo y entrenamiento "
            "inestable. El VAE tiene un marco probabilístico explícito; la GAN optimiza implícitamente la divergencia."
        ),
    },
    {
        "topic": "secuenciales_rnn_lstm_gru",
        "question": "¿Qué puertas usa una GRU y cómo mitigan el problema del gradiente desvaneciente?",
        "ground_truth": (
            "La GRU (Gated Recurrent Unit) usa dos puertas: la puerta de actualización (update gate) que controla "
            "cuánta información del estado anterior se conserva, y la puerta de reset que controla cuánta del estado "
            "anterior se usa para calcular el candidato de estado nuevo. A diferencia de la LSTM, la GRU no tiene "
            "celda de memoria separada. Las puertas aprenden a mantener gradientes no nulos a través del tiempo, "
            "permitiendo capturar dependencias de largo plazo sin el gradiente desvaneciente de la RNN simple."
        ),
    },
    {
        "topic": "explicabilidad",
        "question": "¿Qué mapa produce Grad-CAM y cómo se calcula a partir de los gradientes de la red?",
        "ground_truth": (
            "Grad-CAM (Gradient-weighted Class Activation Mapping) produce un mapa de calor que resalta las regiones "
            "de la imagen más importantes para la predicción de una clase específica. Se calcula obteniendo los "
            "gradientes de la puntuación de la clase respecto a los mapas de activación de la última capa "
            "convolucional, promediando esos gradientes sobre el espacio espacial para obtener pesos por canal, y "
            "luego combinando linealmente los mapas de activación con esos pesos seguido de una ReLU. El resultado "
            "es agnóstico al modelo y no requiere modificar la arquitectura."
        ),
    },
]


def retrieve_contexts(query: str, collection, k: int) -> list[str]:
    expanded = expand_query_for_english_corpus(query)
    ONNXMiniLM_L6_V2.DOWNLOAD_PATH = str(DEFAULT_ONNX_CACHE)
    ef = ONNXMiniLM_L6_V2()
    vector = ef([expanded])[0]
    vector_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)

    result = collection.query(
        query_embeddings=[vector_list],
        n_results=k,
        where={"branch": "deep_learning"},
        include=["documents"],
    )
    return result.get("documents", [[]])[0]


def generate_response(question: str, contexts: list[str], llm: ChatGroq) -> str:
    context_text = "\n\n".join(f"[Contexto {i+1}]\n{c}" for i, c in enumerate(contexts))
    messages = [
        ("system", (
            "Eres un asistente académico experto en Deep Learning. "
            "Responde la pregunta del estudiante basándote ÚNICAMENTE en los contextos proporcionados. "
            "Sé preciso y cita la información de los fragmentos recuperados."
        )),
        ("human", f"Contextos:\n{context_text}\n\nPregunta: {question}"),
    ]
    response = llm.invoke(messages)
    return response.content


def build_samples(collection, k: int, llm: ChatGroq) -> list[SingleTurnSample]:
    samples = []
    for item in EVAL_SUITE:
        print(f"  Evaluando: {item['topic']}...")
        contexts = retrieve_contexts(item["question"], collection, k)
        if not contexts:
            print(f"    Sin contexto para {item['topic']}, saltando.")
            continue
        response = generate_response(item["question"], contexts, llm)
        samples.append(SingleTurnSample(
            user_input=item["question"],
            retrieved_contexts=contexts,
            response=response,
            reference=item["ground_truth"],
        ))
    return samples


def write_report(result, output: Path, k: int) -> None:
    scores = result.to_pandas()
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    lines = [
        "# Reporte de Evaluación RAGAS",
        "",
        f"- Modelo juez: `llama-3.3-70b-versatile` (Groq)",
        f"- Embeddings: `all-MiniLM-L6-v2` (ONNX)",
        f"- Top-k contextos: `{k}`",
        f"- Preguntas evaluadas: `{len(scores)}`",
        "",
        "## Promedios globales",
        "",
        "| Métrica | Valor | Interpretación |",
        "| --- | --- | --- |",
    ]

    interpretations = {
        "faithfulness": "Fidelidad de la respuesta al contexto recuperado (1.0 = perfecta)",
        "answer_relevancy": "Relevancia de la respuesta a la pregunta (1.0 = perfecta)",
        "context_precision": "Precisión del contexto recuperado (proporción útil)",
        "context_recall": "Cobertura del contexto respecto al ground truth",
    }

    for metric in metrics:
        if metric in scores.columns:
            avg = scores[metric].mean()
            lines.append(f"| `{metric}` | {avg:.4f} | {interpretations.get(metric, '')} |")

    lines.extend([
        "",
        "## Detalle por pregunta",
        "",
        "| Tema | " + " | ".join(f"`{m}`" for m in metrics if m in scores.columns) + " |",
        "| --- | " + " | ".join("---" for m in metrics if m in scores.columns) + " |",
    ])

    for i, row in scores.iterrows():
        topic = EVAL_SUITE[i]["topic"] if i < len(EVAL_SUITE) else f"pregunta_{i}"
        values = " | ".join(
            f"{row[m]:.3f}" if m in scores.columns and not __import__('math').isnan(row[m]) else "N/A"
            for m in metrics
            if m in scores.columns
        )
        lines.append(f"| `{topic}` | {values} |")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReporte escrito en: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación RAGAS del pipeline RAG.")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("-k", type=int, default=4)
    args = parser.parse_args()

    if not args.db_dir.exists():
        print(f"Error: No se encontró vector_db en {args.db_dir}.")
        print("Corre primero: python 01_ingesta/ingestion_chroma_direct.py --embedding-backend onnx-minilm")
        sys.exit(1)

    print("Cargando colección ChromaDB...")
    client = chromadb.PersistentClient(path=str(args.db_dir))
    collection = client.get_collection(COLLECTION_NAME)

    print("Configurando LLM juez (Groq) y embeddings de evaluación...")
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    print(f"\nGenerando respuestas para {len(EVAL_SUITE)} preguntas (k={args.k})...")
    samples = build_samples(collection, args.k, llm)

    if not samples:
        print("No se generaron muestras. Verifica que vector_db esté construido.")
        sys.exit(1)

    dataset = EvaluationDataset(samples=samples)

    print("\nEjecutando evaluación RAGAS (esto puede tardar unos minutos)...")
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print("\n=== Resultados ===")
    print(result)

    write_report(result, args.output, args.k)


if __name__ == "__main__":
    main()
