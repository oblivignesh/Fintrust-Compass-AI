"""
rag/retriever.py

Retriever factory with domain-aware filtering.
Used by agents at query time (not during ingestion).
"""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def get_domain_retriever(domain: Optional[str] = None, k: Optional[int] = None):
    """
    Load the persisted ChromaDB and return a retriever.

    Args:
        domain : One of 'loans', 'deposits', 'accounts', 'cards',
                 'compliance', 'digital_banking'. None = global search.
        k      : Number of chunks to retrieve. Defaults to RETRIEVAL_K in .env.
    """
    from ingestion.vector_store import get_retriever, load_vector_store

    retrieval_k = k or int(os.getenv("RETRIEVAL_K", 6))
    vector_store = load_vector_store()
    return get_retriever(vector_store, domain=domain, k=retrieval_k)
