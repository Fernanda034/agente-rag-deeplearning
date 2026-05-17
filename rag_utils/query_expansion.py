from __future__ import annotations

import re
import unicodedata


SPANISH_TO_ENGLISH_TERMS = {
    ("sobreajuste", "memorice", "memorizar", "generalizacion", "regularizacion"): [
        "overfitting",
        "generalization",
        "regularization",
        "dropout",
        "weight decay",
    ],
    ("normalizacion", "lote", "batch normalization", "batchnorm"): [
        "batch normalization",
        "internal covariate shift",
        "mini-batch statistics",
    ],
    ("gradiente", "optimizacion", "descenso", "adam"): [
        "gradient descent",
        "stochastic optimization",
        "Adam optimizer",
        "adaptive learning rates",
    ],
    ("atencion", "autoatencion", "transformer", "multi-head", "multi head"): [
        "attention",
        "self-attention",
        "multi-head attention",
        "transformer",
    ],
    ("convolucion", "cnn", "vision", "resnet", "residual"): [
        "convolutional neural network",
        "residual learning",
        "skip connections",
        "ResNet",
    ],
    ("recurrente", "secuencia", "serie", "gru", "lstm"): [
        "recurrent neural network",
        "gated recurrent unit",
        "LSTM",
        "sequence modeling",
    ],
    ("explicabilidad", "mapa", "saliency", "grad-cam", "interpretabilidad"): [
        "explainability",
        "Grad-CAM",
        "class activation map",
        "visual explanations",
    ],
    ("generativo", "vae", "gan", "autoencoder", "latente"): [
        "generative model",
        "variational autoencoder",
        "generative adversarial network",
        "latent variable",
    ],
}


def normalize_for_matching(text: str) -> str:
    without_accents = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", without_accents.lower()).strip()


def expand_query_for_english_corpus(query: str) -> str:
    normalized = normalize_for_matching(query)
    extra_terms: list[str] = []
    for triggers, terms in SPANISH_TO_ENGLISH_TERMS.items():
        if any(trigger in normalized for trigger in triggers):
            extra_terms.extend(terms)

    if not extra_terms:
        return query

    unique_terms = list(dict.fromkeys(extra_terms))
    return f"{query}\nEnglish technical equivalents: {', '.join(unique_terms)}"
