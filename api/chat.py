"""
API de chat para despliegue en Vercel (Python serverless).

Modos de operación:
  1. RAG completo  — si vector_db/ existe (desarrollo local con ONNX MiniLM + ChromaDB).
  2. Groq directo  — fallback para Vercel (sin vector_db local); usa un system prompt
                     experto en Deep Learning para contestar sin recuperación.

Endpoint:
  POST /api/chat
  Body: {"message": "...", "history": [{"role": "user"|"assistant", "content": "..."}]}
  Response: {"response": "...", "mode": "rag"|"direct"}
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

app = Flask(__name__)
CORS(app)

_agent_executor = None
_agent_load_attempted = False

EXPERT_SYSTEM_PROMPT = """Eres un asistente académico experto en Machine Learning y Deep Learning,
especializado en los siguientes temas del temario universitario:

• Fundamentos: tensores, MLP, funciones de activación, pérdida
• Entrenamiento: SGD, Momentum, RMSProp, Adam, backpropagation
• Estabilidad: dropout, weight decay, BatchNorm, inicialización, gradientes
• Secuenciales: RNN, LSTM, GRU, embeddings, padding, encoder-decoder
• Atención y Transformers: self-attention, multi-head attention, positional encoding, BERT
• Visión: CNN, filtros, stride, ResNet, conexiones residuales, transfer learning
• Explicabilidad: saliency maps, Grad-CAM, visualización de atención
• Generativos: autoencoders, VAE, GANs, espacio latente

Reglas:
1. Responde siempre en español, de manera didáctica y estructurada.
2. Usa LaTeX con signos de dólar para fórmulas (ejemplo: $\\sigma(x) = \\frac{1}{1+e^{-x}}$).
3. Si citas un concepto de un paper conocido, menciona el paper (ej. "Attention Is All You Need, Vaswani et al. 2017").
4. Si no estás seguro de algo, dilo explícitamente.
5. Sé conciso pero completo. Usa listas y secciones para estructurar respuestas largas."""


def _try_load_agent():
    global _agent_executor, _agent_load_attempted
    if _agent_load_attempted:
        return _agent_executor
    _agent_load_attempted = True

    db_path = ROOT_DIR / "vector_db"
    if not db_path.exists():
        return None

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "agent_module", ROOT_DIR / "03_agente" / "agent.py"
        )
        agent_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_mod)
        _agent_executor = agent_mod.get_agent()
        print("Modo RAG completo activado (ChromaDB disponible).")
    except Exception as exc:
        print(f"RAG no disponible, usando Groq directo: {exc}")
        _agent_executor = None

    return _agent_executor


def _respond_with_agent(message: str, history: list[dict]) -> str:
    formatted = []
    for msg in history:
        role = "human" if msg.get("role") == "user" else "assistant"
        formatted.append((role, msg.get("content", "")))

    result = _agent_executor.invoke({"input": message, "chat_history": formatted})
    return result.get("output", "No se pudo generar una respuesta.")


def _respond_direct(message: str, history: list[dict]) -> str:
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

    messages = [SystemMessage(content=EXPERT_SYSTEM_PROMPT)]
    for msg in history[-6:]:  # últimos 6 mensajes para no exceder contexto
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=message))

    response = llm.invoke(messages)
    return response.content


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = data.get("message", "").strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "El campo 'message' es obligatorio."}), 400

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or api_key == "tu_clave_aqui":
        return jsonify({"error": "GROQ_API_KEY no configurada en las variables de entorno."}), 503

    try:
        agent = _try_load_agent()
        if agent is not None:
            response = _respond_with_agent(message, history)
            mode = "rag"
        else:
            response = _respond_direct(message, history)
            mode = "direct"
        return jsonify({"response": response, "mode": mode})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    db_exists = (ROOT_DIR / "vector_db").exists()
    return jsonify({
        "status": "ok",
        "mode": "rag" if db_exists else "direct",
        "groq_key_set": bool(os.environ.get("GROQ_API_KEY")),
    })


if __name__ == "__main__":
    app.run(debug=True, port=8000)
