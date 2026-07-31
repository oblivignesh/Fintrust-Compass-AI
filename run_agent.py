"""
run_agent.py — FinTrust Compass Phase 2 test runner

Interactive CLI to test the multi-agent orchestration pipeline.

Usage:
    python run_agent.py                             # interactive REPL
    python run_agent.py --query "What are home loan eligibility criteria?"
    python run_agent.py --test-all                  # runs a built-in test suite
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Built-in test queries — one per domain
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    ("loans",          "What are the eligibility criteria for a home loan at FinTrust?"),
    ("loans",          "What documents are required for a vehicle loan application?"),
    ("deposits",       "What is the interest rate for a 1-year fixed deposit?"),
    ("deposits",       "What happens if a customer breaks a recurring deposit before maturity?"),
    ("accounts",       "What is the minimum balance requirement for a savings account?"),
    ("accounts",       "What KYC documents are needed to open a current account?"),
    ("cards",          "What are the annual fees for FinTrust credit cards?"),
    ("cards",          "How does the credit card billing cycle work?"),
    ("compliance",     "What are the AML reporting requirements for suspicious transactions?"),
    ("digital_banking","What two-factor authentication methods does FinTrust support for internet banking?"),
]


def print_result(result: dict, query: str) -> None:
    print("\n" + "=" * 70)
    print(f"QUERY   : {query}")
    print(f"DOMAIN  : {result['domain']}  (confidence: {result['confidence']})")
    print("-" * 70)
    print(f"ANSWER  :\n{result['answer']}")
    print("-" * 70)
    print(f"SOURCES : {len(result['sources'])} chunks retrieved")
    for i, src in enumerate(result["sources"][:3], 1):
        print(f"  [{i}] {src['source_file']} | page {src['page']}")
        print(f"       \"{src['snippet'][:100]}...\"")
    print("=" * 70)


def run_interactive():
    """Interactive REPL for the agent."""
    from agents.orchestrator import run_query

    print("\n" + "=" * 70)
    print("  FinTrust Compass — Employee Assistant (Phase 2)")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 70 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if not query:
            continue

        try:
            result = run_query(query)
            print_result(result, query)
        except Exception as e:
            print(f"[ERROR] {e}")


def run_single_query(query: str):
    """Run a single query and print the result."""
    from agents.orchestrator import run_query
    result = run_query(query)
    print_result(result, query)


def run_test_suite():
    """Run all built-in test queries and print a summary."""
    from agents.orchestrator import run_query

    print("\n" + "=" * 70)
    print("  FinTrust Compass — Multi-Agent Test Suite")
    print(f"  Running {len(TEST_QUERIES)} test queries...")
    print("=" * 70)

    results = []
    for expected_domain, query in TEST_QUERIES:
        print(f"\n[TEST] {query[:60]}...")
        result = run_query(query)
        correct = result["domain"] == expected_domain
        results.append(correct)
        status = "PASS" if correct else f"FAIL (expected {expected_domain}, got {result['domain']})"
        print(f"  Domain routing: {status}")
        print(f"  Answer length : {len(result['answer'])} chars")
        print(f"  Sources       : {len(result['sources'])} chunks")

    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 70}")
    print(f"  Routing accuracy: {passed}/{total} ({passed/total*100:.0f}%)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="FinTrust Compass — Phase 2 agent runner")
    parser.add_argument("--query", "-q", type=str, help="Single query to run")
    parser.add_argument("--test-all", action="store_true", help="Run built-in test suite")
    args = parser.parse_args()

    # Validate ChromaDB exists
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    if not Path(chroma_dir).exists():
        print(f"\n[ERROR] ChromaDB not found at '{chroma_dir}'.")
        print("  Run ingestion first:  python ingest.py")
        sys.exit(1)

    if args.test_all:
        run_test_suite()
    elif args.query:
        run_single_query(args.query)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
