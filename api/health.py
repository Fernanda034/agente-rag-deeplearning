"""
GET /api/health
  Response: {"status": "ok", "mode": "rag"|"direct", "groq_key_set": bool}
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]

app = Flask(__name__)


@app.route("/api/health", methods=["GET"])
def health():
    db_exists = (ROOT_DIR / "vector_db").exists()
    return jsonify({
        "status":       "ok",
        "mode":         "rag" if db_exists else "direct",
        "groq_key_set": bool(os.environ.get("GROQ_API_KEY")),
    })


if __name__ == "__main__":
    app.run(debug=True, port=8001)
