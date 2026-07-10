"""Chunk data/documents/*, embed with BAAI/bge-m3, and load into ChromaDB.

Run with the project's root virtualenv:
    source .venv/bin/activate
    python data/ingest.py
"""

from pathlib import Path

import chromadb
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_PATH = DATA_DIR / "chroma"

COLLECTION_NAME = "job_market_docs"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

_encoding = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_encoding.encode(text))


def load_documents() -> list[tuple[str, str]]:
    """Return [(doc_id, full_text), ...] for every .md/.txt file in data/documents/."""
    paths = sorted(DOCUMENTS_DIR.glob("*.md")) + sorted(DOCUMENTS_DIR.glob("*.txt"))
    return [(path.stem, path.read_text(encoding="utf-8")) for path in paths]


def chunk_documents(docs: list[tuple[str, str]]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=_token_len,
    )
    chunks = []
    for doc_id, text in docs:
        for i, chunk_text in enumerate(splitter.split_text(text)):
            chunks.append({"id": f"{doc_id}::chunk-{i}", "text": chunk_text, "doc_id": doc_id})
    return chunks


def main() -> None:
    docs = load_documents()
    if not docs:
        print(f"No documents found in {DOCUMENTS_DIR}")
        return

    chunks = chunk_documents(docs)

    print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        [c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True
    )
    embedding_dim = embeddings.shape[1]

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[{"source_doc_id": c["doc_id"]} for c in chunks],
    )

    print("\nIngestion complete:")
    print(f"  documents:         {len(docs)}")
    print(f"  chunks:            {len(chunks)}")
    print(f"  embedding dim:     {embedding_dim}")
    print(f"  chroma path:       {CHROMA_PATH}")
    print(f"  collection name:   {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
