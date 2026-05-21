from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.embeddings import Embeddings

ROOT_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_CACHE = ROOT_DIR / ".chroma_onnx_cache" / "onnx_models" / "all-MiniLM-L6-v2"


class ChromaONNXEmbeddings(Embeddings):
    """LangChain-compatible wrapper around ChromaDB's bundled ONNX MiniLM-L6-v2.

    Uses the same ONNX runtime as ingestion_chroma_direct.py so query vectors
    are produced by the identical model implementation as the stored document vectors.
    """

    def __init__(self, cache_dir: str | Path = _DEFAULT_CACHE) -> None:
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
        ONNXMiniLM_L6_V2.DOWNLOAD_PATH = str(cache_dir)
        self._fn = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._fn(texts)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]