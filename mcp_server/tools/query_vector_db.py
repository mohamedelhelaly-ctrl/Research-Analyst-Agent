"""query_vector_db tool: semantic search against the ChromaDB collection
built by data/ingest.py.
"""

from pathlib import Path

import chromadb

from tools.embeddings import embed_query

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_PATH = REPO_ROOT / "data" / "chroma"
COLLECTION_NAME = "job_market_docs"

_client = None
_collection = None


class VectorDBError(Exception):
    """Raised when the ChromaDB collection is unavailable or a query fails."""


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        try:
            _collection = _client.get_collection(COLLECTION_NAME)
        except Exception as exc:
            raise VectorDBError(
                f"Collection '{COLLECTION_NAME}' not found at {CHROMA_PATH}. "
                "Run data/ingest.py first."
            ) from exc
    return _collection


def query_vector_db(query: str, top_k: int = 5) -> list[dict]:
    """Embed `query` and return the top_k most similar chunks."""
    try:
        collection = _get_collection()
        query_embedding = embed_query(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    except VectorDBError:
        raise
    except Exception as exc:
        raise VectorDBError(f"Vector DB query failed: {exc}") from exc

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "chunk_text": doc_text,
            "source_doc_id": metadata.get("source_doc_id"),
            "score": 1 - distance,
        }
        for doc_text, metadata, distance in zip(documents, metadatas, distances)
    ]
