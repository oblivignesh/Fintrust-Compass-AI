"""
ingest.py — FinTrust Compass Phase 1 runner

Run this script once to:
  1. Parse all PDFs in Knowledge_Source/
  2. Split them into overlapping chunks
  3. Embed with Gemini text-embedding-004
  4. Persist into ChromaDB

Usage:
    python ingest.py                   # uses values from .env
    python ingest.py --chunk-size 800  # override chunk size
    python ingest.py --inspect         # show collection summary after ingestion
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on the path so sibling packages resolve
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinTrust Compass — Phase 1 ingestion pipeline")
    parser.add_argument(
        "--pdf-dir",
        default=os.getenv("KNOWLEDGE_SOURCE_DIR", "./Knowledge_Source"),
        help="Directory containing the PDF knowledge source (default: ./Knowledge_Source)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", 1000)),
        help="Maximum characters per chunk (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", 150)),
        help="Overlap between consecutive chunks (default: 150)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print collection summary after ingestion",
    )
    return parser.parse_args()


def validate_env() -> None:
    """Fail fast with a clear message if the API key is missing."""
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n[ERROR] GOOGLE_API_KEY is not set.")
        print("  1. Copy .env.example to .env")
        print("  2. Paste your Google AI Studio key into .env")
        print("  3. Re-run this script.\n")
        sys.exit(1)


def main() -> None:
    args = parse_args()
    validate_env()

    print("\n" + "=" * 60)
    print("  FinTrust Compass — Phase 1: RAG Ingestion Pipeline")
    print("=" * 60)

    start_time = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load & split PDFs
    # ------------------------------------------------------------------
    from ingestion.pdf_loader import load_and_split

    chunks = load_and_split(
        pdf_dir=args.pdf_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if not chunks:
        print("[ERROR] No chunks were produced. Check the PDF directory.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Embed & persist in ChromaDB
    # ------------------------------------------------------------------
    from ingestion.vector_store import build_vector_store, inspect_collection

    vector_store = build_vector_store(chunks)

    # ------------------------------------------------------------------
    # Step 3: Optional inspection
    # ------------------------------------------------------------------
    if args.inspect:
        inspect_collection(vector_store)

    # ------------------------------------------------------------------
    # Step 4: Quick smoke test — retrieve 3 chunks for a sample query
    # ------------------------------------------------------------------
    print("\n[Smoke Test] Running a test retrieval query...")
    from ingestion.vector_store import get_retriever

    retriever = get_retriever(vector_store, k=3)
    test_query = "What are the eligibility criteria for a home loan?"
    results = retriever.invoke(test_query)

    print(f"\nQuery: \"{test_query}\"")
    print(f"Retrieved {len(results)} chunks:\n")
    for i, doc in enumerate(results, 1):
        meta = doc.metadata
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"  [{i}] {meta.get('source_file', 'unknown')} | "
              f"domain={meta.get('domain')} | page={meta.get('page', '?')}")
        print(f"       \"{snippet}...\"\n")

    elapsed = time.time() - start_time
    print(f"{'='*60}")
    print(f"  Ingestion complete in {elapsed:.1f}s")
    print(f"  ChromaDB persisted at: {os.getenv('CHROMA_PERSIST_DIR', './chroma_db')}")
    print(f"  Run 'python ingest.py --inspect' to review domain distribution.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
