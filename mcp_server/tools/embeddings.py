"""Shared BAAI/bge-m3 embedding model loader for MCP server tools.

Kept in sync manually with data/ingest.py, which embeds documents with the
same model and settings when building the ChromaDB collection.
"""

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_query(text: str) -> list[float]:
    model = get_model()
    return model.encode([text], normalize_embeddings=True)[0].tolist()
