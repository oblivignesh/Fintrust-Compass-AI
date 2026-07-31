"""
ingestion/embedder.py

Rate-limit-safe Gemini embedding adapter with retry.
"""

import os
import time
import warnings
from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

warnings.filterwarnings("ignore")
load_dotenv()

_API_BATCH_SIZE = 20      # texts per embed_content call (paid-tier safe)
_INTER_BATCH_SLEEP = 0.3  # seconds between calls (paid tier can sustain faster)
_MAX_RETRIES = 8          # retry on transient errors


class GeminiBatchEmbeddings(Embeddings):
    """
    Wrapper that sends texts in small batches with inter-call sleep and
    exponential-backoff retry to stay within Gemini Embedding rate limits.
    """

    def __init__(self, model: str, api_key: str, task_type: str = "retrieval_document"):
        self._model = model
        self._api_key = api_key
        self._doc_embedder = GoogleGenerativeAIEmbeddings(
            model=model, google_api_key=api_key, task_type=task_type
        )
        self._query_embedder = GoogleGenerativeAIEmbeddings(
            model=model, google_api_key=api_key, task_type="retrieval_query"
        )

    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed a small batch with exponential backoff on failure."""
        delay = 5.0
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return self._doc_embedder.embed_documents(texts)
            except Exception as e:
                err_msg = str(e)
                if attempt == _MAX_RETRIES:
                    raise
                print(f"    [Retry {attempt}/{_MAX_RETRIES}] Error: {err_msg[:120]}. Sleeping {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, 60)
        return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        total = len(texts)
        total_batches = (total + _API_BATCH_SIZE - 1) // _API_BATCH_SIZE
        for idx, start in enumerate(range(0, total, _API_BATCH_SIZE)):
            batch = texts[start : start + _API_BATCH_SIZE]
            print(f"    sub-batch {idx+1}/{total_batches} ({len(batch)} texts)...")
            result = self._embed_with_retry(batch)
            all_embeddings.extend(result)
            if start + _API_BATCH_SIZE < total:
                time.sleep(_INTER_BATCH_SLEEP)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        delay = 5.0
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return self._query_embedder.embed_query(text)
            except Exception as e:
                if attempt == _MAX_RETRIES:
                    raise
                print(f"    [Retry {attempt}/{_MAX_RETRIES}] Query embed error. Sleeping {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, 60)
        return []


def _get_model_and_key() -> tuple:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    return model_name, api_key


@lru_cache(maxsize=1)
def get_embeddings() -> GeminiBatchEmbeddings:
    """Return a cached document-mode embeddings instance."""
    model_name, api_key = _get_model_and_key()
    print(f"[Embedder] Model: {model_name}  batch_size={_API_BATCH_SIZE}")
    return GeminiBatchEmbeddings(model=model_name, api_key=api_key, task_type="retrieval_document")


def get_query_embeddings() -> GeminiBatchEmbeddings:
    """Return a query-optimised embeddings instance (used at retrieval time)."""
    model_name, api_key = _get_model_and_key()
    return GeminiBatchEmbeddings(model=model_name, api_key=api_key, task_type="retrieval_query")
