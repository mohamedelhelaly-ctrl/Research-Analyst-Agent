"""read_document tool: reads a document from the local corpus by doc_id."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCUMENTS_DIR = REPO_ROOT / "data" / "documents"
MAX_CHARS = 8000


class DocumentNotFoundError(Exception):
    """Raised when no document matches the given doc_id."""


def read_document(doc_id: str) -> str:
    for ext in (".md", ".txt"):
        path = DOCUMENTS_DIR / f"{doc_id}{ext}"
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if len(text) > MAX_CHARS:
                return (
                    text[:MAX_CHARS]
                    + f"\n\n[... truncated; original length {len(text)} characters ...]"
                )
            return text
    raise DocumentNotFoundError(
        f"No document found for doc_id='{doc_id}' in {DOCUMENTS_DIR}"
    )
