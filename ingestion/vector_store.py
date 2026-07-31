"""
ingestion/vector_store.py

Builds and manages the ChromaDB vector store for FinTrust Compass.

Key design decisions:
- Single ChromaDB collection "fintrust_policies" holds all document chunks.
- Every chunk carries a 'domain' metadata field so individual agents can
  perform filtered retrieval (e.g., loans agent only sees loan chunks).
- Persistence is enabled so the DB survives restarts without re-ingestion.
"""

import os
from pathlib import Path
from typing import List, Optional

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from ingestion.embedder import GeminiBatchEmbeddings, get_embeddings, get_query_embeddings

load_dotenv()

COLLECTION_NAME = "fintrust_policies"


def _get_persist_dir() -> str:
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return persist_dir


# ---------------------------------------------------------------------------
# Build (first-time ingestion)
# ---------------------------------------------------------------------------

def build_vector_store(chunks: List[Document]) -> Chroma:
    """
    Embed *chunks* using the Gemini embedding model and persist them in
    ChromaDB.

    RESUMABLE: If ChromaDB already exists from a previous (interrupted) run,
    this function checks which chunk_indexes are already stored and skips
    them — only embedding the remaining chunks. Safe to call multiple times.

    Args:
        chunks: List of LangChain Document objects with metadata.

    Returns:
        A ready-to-query Chroma vector store instance.
    """
    persist_dir = _get_persist_dir()
    embeddings = get_embeddings()

    # ------------------------------------------------------------------
    # Determine which chunks are already stored (resume support)
    # ------------------------------------------------------------------
    chroma_path = Path(persist_dir)
    already_stored: set = set()
    vector_store = None

    if chroma_path.exists() and any(chroma_path.iterdir()):
        try:
            vector_store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=persist_dir,
            )
            existing_count = vector_store._collection.count()
            if existing_count > 0:
                sample = vector_store._collection.get(
                    limit=existing_count, include=["metadatas"]
                )
                already_stored = {
                    m.get("chunk_index", -1)
                    for m in sample["metadatas"]
                    if m.get("chunk_index", -1) >= 0
                }
                print(f"\n[Vector Store] Resuming — {len(already_stored)} chunks already stored, "
                      f"{len(chunks) - len(already_stored)} remaining.")
        except Exception:
            already_stored = set()

    # Filter to only chunks that need embedding
    remaining = [c for c in chunks if c.metadata.get("chunk_index", -1) not in already_stored]

    if not remaining:
        print("[Vector Store] All chunks already embedded. Nothing to do.")
        return vector_store  # type: ignore[return-value]

    print(f"\n[Vector Store] Building ChromaDB at: {persist_dir}")
    print(f"[Vector Store] Embedding {len(remaining)} chunks (this may take a while)...")

    # Batch in groups of 100; the embedder internally splits into sub-batches of 20.
    batch_size = 100
    total_batches = (len(remaining) + batch_size - 1) // batch_size

    for i in range(total_batches):
        batch = remaining[i * batch_size : (i + 1) * batch_size]
        print(f"  Batch {i + 1}/{total_batches} — {len(batch)} chunks "
              f"(indexes {batch[0].metadata.get('chunk_index')}–{batch[-1].metadata.get('chunk_index')})")

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=persist_dir,
            )
        else:
            vector_store.add_documents(batch)

    doc_count = vector_store._collection.count()
    print(f"\n[Vector Store] Ingestion complete. Total vectors stored: {doc_count}")
    return vector_store


# ---------------------------------------------------------------------------
# Load (subsequent runs — no re-embedding)
# ---------------------------------------------------------------------------

def load_vector_store() -> Chroma:
    """
    Load an existing persisted ChromaDB vector store.
    Raises FileNotFoundError if the store has not been built yet.
    """
    persist_dir = _get_persist_dir()

    if not Path(persist_dir).exists() or not any(Path(persist_dir).iterdir()):
        raise FileNotFoundError(
            f"ChromaDB not found at '{persist_dir}'. "
            "Run the ingestion pipeline first: python ingest.py"
        )

    print(f"[Vector Store] Loading existing ChromaDB from: {persist_dir}")
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    doc_count = vector_store._collection.count()
    print(f"[Vector Store] Loaded {doc_count} vectors.")
    return vector_store


# ---------------------------------------------------------------------------
# Retriever factory
# ---------------------------------------------------------------------------

def get_retriever(
    vector_store: Chroma,
    domain: Optional[str] = None,
    k: int = 6,
) -> VectorStoreRetriever:
    """
    Return a LangChain retriever from the vector store.

    Args:
        vector_store: A loaded or built Chroma instance.
        domain: If provided, restricts retrieval to chunks with this domain
                tag (e.g. 'loans', 'cards', 'deposits', 'compliance',
                'digital_banking', 'accounts').
        k: Number of top chunks to retrieve per query.

    Returns:
        A LangChain VectorStoreRetriever.
    """
    search_kwargs: dict = {"k": k}

    if domain:
        search_kwargs["filter"] = {"domain": domain}
        print(f"[Retriever] Domain-filtered retriever: domain='{domain}', k={k}")
    else:
        print(f"[Retriever] Global retriever: k={k}")

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def inspect_collection(vector_store: Chroma) -> None:
    """Print a summary of what's stored in the ChromaDB collection."""
    collection = vector_store._collection
    total = collection.count()
    print(f"\n{'='*50}")
    print(f"ChromaDB Collection: {COLLECTION_NAME}")
    print(f"Total vectors: {total}")

    # Sample 10 records and show domain distribution
    sample = collection.get(limit=total, include=["metadatas"])
    domains: dict = {}
    products: dict = {}
    for meta in sample["metadatas"]:
        d = meta.get("domain", "unknown")
        p = meta.get("product", "unknown")
        domains[d] = domains.get(d, 0) + 1
        products[p] = products.get(p, 0) + 1

    print("\nChunks by domain:")
    for domain, count in sorted(domains.items()):
        print(f"  {domain:<20} {count} chunks")

    print("\nChunks by product:")
    for product, count in sorted(products.items()):
        print(f"  {product:<30} {count} chunks")
    print(f"{'='*50}\n")
