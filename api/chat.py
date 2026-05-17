"""
POST /api/chat
  Body:     {"message": "...", "history": [{"role": "user"|"assistant", "content": "..."}]}
  Response: {"response": "...", "sources": [...], "mode": "rag"|"direct"}

Modes:
  rag    — vector_db/ found locally: full ChromaDB + LangChain agent pipeline.
  direct — no vector_db (Vercel deploy): Groq Llama 3.3 70b with expert system prompt.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

app = Flask(__name__)
CORS(app)

# ── Serve public/index.html for local dev (python api/chat.py) ───────────────
@app.route("/")
def index():
    return send_from_directory(str(ROOT_DIR / "public"), "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(ROOT_DIR / "public"), filename)


# ── Expert system prompt (direct / Groq-only mode) ───────────────────────────
_SYSTEM_PROMPT = """\
Eres un asistente académico experto en Machine Learning y Deep Learning, \
diseñado para ayudar a estudiantes universitarios con su temario.

Temas que dominas:
• Fundamentos: tensores, MLP, funciones de activación, función de pérdida
• Entrenamiento: SGD, Momentum, RMSProp, Adam, backpropagation, schedulers
• Regularización: dropout, weight decay, BatchNorm, early stopping, inicialización
• Secuenciales: RNN, LSTM, GRU, embeddings, tokenización, encoder-decoder
• Atención y Transformers: self-attention, multi-head attention, positional encoding, BERT
• Visión: CNN, filtros, stride, padding, ResNet, transfer learning, data augmentation
• Explicabilidad: saliency maps, Grad-CAM, class activation maps
• Generativos: autoencoders, VAE, GANs, espacio latente, estabilidad

Reglas obligatorias:
1. Responde siempre en español, de manera didáctica y estructurada.
2. Escribe fórmulas en LaTeX entre signos de dólar: $E = mc^2$ o $$\\sigma(x) = \\frac{1}{1+e^{-x}}$$.
3. Cita el paper fuente cuando lo conozcas (ej. "Attention Is All You Need, Vaswani et al. 2017").
4. Usa listas, secciones y negrita para estructurar respuestas largas.
5. Si no estás seguro, dilo explícitamente.\
"""

# ── Lazy-load the RAG agent (only when vector_db/ exists) ─────────────────────
_agent = None
_agent_attempted = False


def _load_agent():
    global _agent, _agent_attempted
    if _agent_attempted:
        return _agent
    _agent_attempted = True

    db_path = ROOT_DIR / "vector_db"
    if not db_path.exists():
        return None

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "agent_mod", ROOT_DIR / "03_agente" / "agent.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _agent = mod.get_agent()
        print("✓ RAG mode: ChromaDB loaded.")
    except Exception as exc:
        print(f"✗ RAG unavailable, falling back to direct: {exc}")
        _agent = None

    return _agent


# ── Source extraction via LangChain callback ──────────────────────────────────
class _SourceCapture:
    """Minimal retriever callback that collects document metadata."""
    def __init__(self):
        self.sources: list[dict] = []

    # LangChain calls this after each retriever invocation
    def on_retriever_end(self, documents, **_):
        seen = set()
        for doc in documents:
            m = doc.metadata
            key = (m.get("file_name", ""), m.get("page", ""))
            if key in seen or not key[0]:
                continue
            seen.add(key)
            self.sources.append({
                "file":  m.get("file_name", m.get("source", "Desconocido")),
                "page":  m.get("page"),
                "topic": m.get("topic", ""),
            })


# ── Response helpers ──────────────────────────────────────────────────────────
def _rag_response(message: str, history: list[dict]) -> tuple[str, list[dict]]:
    capture = _SourceCapture()

    # Format history as LangChain expects
    formatted = [
        ("human" if m["role"] == "user" else "assistant", m["content"])
        for m in history
    ]

    result = _agent.invoke(
        {"input": message, "chat_history": formatted},
        config={"callbacks": [capture]},
    )
    return result.get("output", "Sin respuesta."), capture.sources


def _direct_response(message: str, history: list[dict]) -> tuple[str, list[dict]]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

    msgs = [SystemMessage(content=_SYSTEM_PROMPT)]
    for m in history[-8:]:  # keep last 8 turns to stay within context window
        cls = HumanMessage if m["role"] == "user" else AIMessage
        msgs.append(cls(content=m["content"]))
    msgs.append(HumanMessage(content=message))

    return llm.invoke(msgs).content, []


# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    data    = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "").strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "El campo 'message' es obligatorio."}), 400

    if not os.environ.get("GROQ_API_KEY"):
        return jsonify({"error": "GROQ_API_KEY no está configurada."}), 503

    try:
        agent = _load_agent()
        if agent is not None:
            response, sources = _rag_response(message, history)
            mode = "rag"
        else:
            response, sources = _direct_response(message, history)
            mode = "direct"
        return jsonify({"response": response, "sources": sources, "mode": mode})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting dev server at http://localhost:{port}")
    app.run(debug=True, port=port)
