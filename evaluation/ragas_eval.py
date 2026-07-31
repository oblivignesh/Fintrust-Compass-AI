"""
evaluation/ragas_eval.py

RAGAS-style evaluation for FinTrust Compass.

Implements 4 core RAGAS metrics using Gemini 2.5 Flash as the judge LLM:
  1. Faithfulness         — answer uses only information from retrieved contexts
  2. Answer Relevancy     — answer actually addresses the question
  3. Context Precision    — retrieved chunks are relevant to the question
  4. Context Recall       — retrieved chunks cover the ground truth

Usage:
    python -W ignore evaluation/ragas_eval.py            # evaluate all 24 samples
    python -W ignore evaluation/ragas_eval.py --limit 6  # quick smoke test (1 per domain)
    python -W ignore evaluation/ragas_eval.py --id 3     # evaluate single sample by id
"""

import sys
import json
import time
import argparse
import textwrap
from pathlib import Path
from typing import Any

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from langchain_google_genai import ChatGoogleGenerativeAI
from agents.orchestrator import build_graph
from ingestion.vector_store import load_vector_store, get_retriever

# ── judge LLM ─────────────────────────────────────────────────────────────────
_judge: ChatGoogleGenerativeAI | None = None

def _get_judge() -> ChatGoogleGenerativeAI:
    global _judge
    if _judge is None:
        _judge = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return _judge


def _ask_judge(prompt: str, retries: int = 4, base_sleep: float = 5.0) -> str:
    """Call the judge LLM with simple exponential-backoff retry."""
    judge = _get_judge()
    for attempt in range(retries):
        try:
            return judge.invoke(prompt).content.strip()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            sleep_s = base_sleep * (2 ** attempt)
            print(f"      [retry {attempt+1}/{retries}] {exc} — sleeping {sleep_s:.0f}s")
            time.sleep(sleep_s)
    return ""


# ── metric implementations ────────────────────────────────────────────────────

def score_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """
    Faithfulness: fraction of answer claims that are supported by the contexts.
    Score 0-1 (higher is better).
    """
    ctx_block = "\n---\n".join(contexts[:6])
    prompt = textwrap.dedent(f"""
        You are an expert evaluator for a banking question-answering system.

        QUESTION: {question}

        RETRIEVED CONTEXT:
        {ctx_block}

        GENERATED ANSWER:
        {answer}

        Task: Assess whether every factual claim in the GENERATED ANSWER is supported
        by the RETRIEVED CONTEXT. Ignore claims that are common sense or definitions.

        Respond with ONLY a JSON object in this exact format:
        {{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}

        where score = 1.0 means every claim is fully supported,
                      0.0 means the answer contains unsupported fabrications.
    """).strip()
    try:
        raw = _ask_judge(prompt)
        raw = raw.replace("```json", "").replace("```", "").strip()
        return float(json.loads(raw)["score"])
    except Exception as e:
        print(f"      [faithfulness parse error] {e} raw={raw[:120]!r}")
        return 0.0


def score_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy: how well the answer addresses the question.
    Score 0-1.
    """
    prompt = textwrap.dedent(f"""
        You are an expert evaluator for a banking question-answering system.

        QUESTION: {question}

        GENERATED ANSWER:
        {answer}

        Task: Rate how directly and completely the GENERATED ANSWER addresses the QUESTION.
        A score of 1.0 means the answer fully answers the question.
        A score of 0.0 means the answer is irrelevant or off-topic.

        Respond with ONLY a JSON object:
        {{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}
    """).strip()
    try:
        raw = _ask_judge(prompt)
        raw = raw.replace("```json", "").replace("```", "").strip()
        return float(json.loads(raw)["score"])
    except Exception as e:
        print(f"      [answer_relevancy parse error] {e} raw={raw[:120]!r}")
        return 0.0


def score_context_precision(question: str, contexts: list[str]) -> float:
    """
    Context Precision: fraction of retrieved chunks that are relevant to the question.
    Score 0-1.
    """
    if not contexts:
        return 0.0

    relevance_scores = []
    for i, ctx in enumerate(contexts[:6]):
        prompt = textwrap.dedent(f"""
            You are an expert evaluator for a banking question-answering system.

            QUESTION: {question}

            RETRIEVED CHUNK:
            {ctx[:800]}

            Task: Is this chunk useful for answering the question?
            Respond with ONLY a JSON object:
            {{"relevant": <true|false>, "reason": "<one sentence>"}}
        """).strip()
        try:
            raw = _ask_judge(prompt)
            raw = raw.replace("```json", "").replace("```", "").strip()
            relevance_scores.append(1.0 if json.loads(raw)["relevant"] else 0.0)
        except Exception as e:
            print(f"      [context_precision parse error chunk {i}] {e}")
            relevance_scores.append(0.0)

    return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0


def score_context_recall(question: str, ground_truth: str, contexts: list[str]) -> float:
    """
    Context Recall: fraction of ground-truth claims covered by the retrieved contexts.
    Score 0-1.
    """
    ctx_block = "\n---\n".join(contexts[:6])
    prompt = textwrap.dedent(f"""
        You are an expert evaluator for a banking question-answering system.

        QUESTION: {question}

        GROUND TRUTH ANSWER:
        {ground_truth}

        RETRIEVED CONTEXT:
        {ctx_block}

        Task: Assess what fraction of the factual claims in the GROUND TRUTH ANSWER
        can be found in the RETRIEVED CONTEXT.

        Respond with ONLY a JSON object:
        {{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}

        where 1.0 = all ground truth claims are present in the context,
              0.0 = none of the ground truth claims appear in the context.
    """).strip()
    try:
        raw = _ask_judge(prompt)
        raw = raw.replace("```json", "").replace("```", "").strip()
        return float(json.loads(raw)["score"])
    except Exception as e:
        print(f"      [context_recall parse error] {e} raw={raw[:120]!r}")
        return 0.0


# ── evaluation runner ─────────────────────────────────────────────────────────

def evaluate_sample(graph: Any, vs: Any, sample: dict) -> dict:
    """Run one sample through the pipeline and score all 4 metrics."""
    question = sample["question"]
    ground_truth = sample["ground_truth"]
    expected_domain = sample["domain"]

    print(f"\n  Q: {question[:80]}")

    # Run through orchestrator
    result = graph.invoke({"query": question, "messages": []})
    answer = result.get("answer", "")
    actual_domain = result.get("domain", "unknown")
    confidence = result.get("confidence", "unknown")
    sources = result.get("sources", [])

    # Retrieve FULL document content for accurate metric scoring
    # (sources only store 200-char snippets; we need full text for judge)
    retriever = get_retriever(vs, domain=actual_domain if actual_domain != "unknown" else None, k=6)
    full_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in full_docs]

    print(f"     domain={actual_domain} ({confidence}), sources={len(sources)}, ctx_chars={sum(len(c) for c in contexts)}")
    print(f"     answer[:100]={answer[:100]!r}")

    # Score all 4 metrics (with a short pause between judge calls to avoid rate limits)
    print("     [scoring faithfulness]")
    faith = score_faithfulness(question, answer, contexts)
    time.sleep(1)

    print("     [scoring answer_relevancy]")
    rel = score_answer_relevancy(question, answer)
    time.sleep(1)

    print("     [scoring context_precision]")
    prec = score_context_precision(question, contexts)
    time.sleep(1)

    print("     [scoring context_recall]")
    recall = score_context_recall(question, ground_truth, contexts)
    time.sleep(1)

    domain_correct = int(actual_domain == expected_domain)

    return {
        "id": sample["id"],
        "domain": expected_domain,
        "actual_domain": actual_domain,
        "domain_correct": domain_correct,
        "confidence": confidence,
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "faithfulness": round(faith, 3),
        "answer_relevancy": round(rel, 3),
        "context_precision": round(prec, 3),
        "context_recall": round(recall, 3),
        "ragas_score": round((faith + rel + prec + recall) / 4, 3),
    }


def print_results_table(results: list[dict]) -> None:
    domains = sorted({r["domain"] for r in results})

    print("\n" + "=" * 90)
    print("FINTRUST COMPASS — RAGAS EVALUATION RESULTS")
    print("=" * 90)

    header = f"{'ID':>3}  {'Domain':<16} {'DomOK':>5}  {'Faith':>5}  {'Relev':>5}  {'Prec':>5}  {'Recall':>6}  {'RAGAS':>6}"
    print(header)
    print("-" * 90)

    for r in results:
        ok_icon = "✅" if r["domain_correct"] else "❌"
        print(
            f"{r['id']:>3}  {r['domain']:<16} {ok_icon:>5}  "
            f"{r['faithfulness']:>5.3f}  {r['answer_relevancy']:>5.3f}  "
            f"{r['context_precision']:>5.3f}  {r['context_recall']:>6.3f}  "
            f"{r['ragas_score']:>6.3f}"
        )

    print("-" * 90)

    # Per-domain averages
    print("\nPER-DOMAIN AVERAGES")
    print(f"{'Domain':<18} {'N':>3}  {'DomAcc':>6}  {'Faith':>5}  {'Relev':>5}  {'Prec':>5}  {'Recall':>6}  {'RAGAS':>6}")
    print("-" * 65)
    for d in domains:
        dr = [r for r in results if r["domain"] == d]
        n = len(dr)
        print(
            f"{d:<18} {n:>3}  "
            f"{sum(r['domain_correct'] for r in dr)/n:>6.2%}  "
            f"{sum(r['faithfulness'] for r in dr)/n:>5.3f}  "
            f"{sum(r['answer_relevancy'] for r in dr)/n:>5.3f}  "
            f"{sum(r['context_precision'] for r in dr)/n:>5.3f}  "
            f"{sum(r['context_recall'] for r in dr)/n:>6.3f}  "
            f"{sum(r['ragas_score'] for r in dr)/n:>6.3f}"
        )

    # Overall
    n = len(results)
    print("-" * 65)
    print(
        f"{'OVERALL':<18} {n:>3}  "
        f"{sum(r['domain_correct'] for r in results)/n:>6.2%}  "
        f"{sum(r['faithfulness'] for r in results)/n:>5.3f}  "
        f"{sum(r['answer_relevancy'] for r in results)/n:>5.3f}  "
        f"{sum(r['context_precision'] for r in results)/n:>5.3f}  "
        f"{sum(r['context_recall'] for r in results)/n:>6.3f}  "
        f"{sum(r['ragas_score'] for r in results)/n:>6.3f}"
    )
    print("=" * 90)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FinTrust Compass — RAGAS Evaluation")
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only the first N samples (0 = all)")
    parser.add_argument("--id", type=int, default=0,
                        help="Evaluate a single sample by its id")
    parser.add_argument("--output", type=str,
                        default=str(ROOT / "evaluation" / "results.json"),
                        help="Path to save JSON results")
    args = parser.parse_args()

    dataset_path = ROOT / "evaluation" / "test_dataset.json"
    with open(dataset_path) as f:
        dataset = json.load(f)

    if args.id:
        dataset = [s for s in dataset if s["id"] == args.id]
        if not dataset:
            print(f"[ERROR] No sample with id={args.id}")
            sys.exit(1)
    elif args.limit:
        dataset = dataset[: args.limit]

    print(f"\n[FinTrust Eval] Loading orchestrator graph...")
    graph = build_graph()
    print(f"[FinTrust Eval] Loading ChromaDB for full-context retrieval...")
    vs = load_vector_store()

    print(f"[FinTrust Eval] Evaluating {len(dataset)} sample(s)...\n")
    results = []
    for i, sample in enumerate(dataset, 1):
        print(f"[{i}/{len(dataset)}] id={sample['id']} domain={sample['domain']}")
        try:
            row = evaluate_sample(graph, vs, sample)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            row = {
                "id": sample["id"], "domain": sample["domain"],
                "actual_domain": "ERROR", "domain_correct": 0,
                "question": sample["question"], "answer": "", "ground_truth": sample["ground_truth"],
                "faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "ragas_score": 0.0,
            }
        results.append(row)

    print_results_table(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[FinTrust Eval] Full results saved → {args.output}")


if __name__ == "__main__":
    main()
