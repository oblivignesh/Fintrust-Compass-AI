"""
agents/comparison_agent.py

Policy Comparison Tool for FinTrust Compass.

LangGraph flow:
    [START] -> [retrieve_both] -> [extract_parameters] -> [build_comparison] -> [END]

Given two products, the agent:
  1. Retrieves policy chunks for BOTH products in parallel from the vector store
  2. Asks Gemini to extract a consistent set of comparable parameters for each
  3. Builds a structured side-by-side comparison table + a recommendation summary

Supported product pairs come from COMPARABLE_GROUPS — only products in the
same group make sense to compare (e.g., Home Loan vs Personal Loan, FD vs RD).
Cross-group comparisons (e.g., Loan vs Card) are also allowed for flexibility.
"""

from __future__ import annotations
import json, os, warnings
from typing import Optional
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from ingestion.vector_store import get_retriever, load_vector_store

load_dotenv()

# ── Product catalogue ─────────────────────────────────────────────────────────

COMPARE_PRODUCT_CONFIG: dict[str, dict] = {
    "home_loan":         {"domain": "loans",    "label": "Home Loan",
                          "search_query": "home loan interest rate tenure eligibility processing fee prepayment foreclosure"},
    "personal_loan":     {"domain": "loans",    "label": "Personal Loan",
                          "search_query": "personal loan interest rate tenure eligibility processing fee prepayment foreclosure"},
    "vehicle_loan":      {"domain": "loans",    "label": "Vehicle Loan",
                          "search_query": "vehicle loan interest rate tenure eligibility processing fee prepayment LTV"},
    "fixed_deposit":     {"domain": "deposits", "label": "Fixed Deposit",
                          "search_query": "fixed deposit interest rate minimum deposit tenure compounding premature withdrawal TDS"},
    "recurring_deposit": {"domain": "deposits", "label": "Recurring Deposit",
                          "search_query": "recurring deposit interest rate minimum installment tenure premature withdrawal"},
    "savings_account":   {"domain": "accounts", "label": "Savings Account",
                          "search_query": "savings account minimum balance interest rate features charges withdrawal limit"},
    "current_account":   {"domain": "accounts", "label": "Current Account",
                          "search_query": "current account minimum balance charges overdraft features cash deposit limit"},
    "salary_account":    {"domain": "accounts", "label": "Salary Account",
                          "search_query": "salary account features zero balance interest charges benefits"},
    "credit_card":       {"domain": "cards",    "label": "Credit Card",
                          "search_query": "credit card interest rate annual fee reward points credit limit cash advance charges"},
    "debit_card":        {"domain": "cards",    "label": "Debit Card",
                          "search_query": "debit card annual fee ATM charges daily limit features cashback"},
}

# Logical groupings shown as quick-pick pairs in the UI
COMPARABLE_GROUPS: list[dict] = [
    {"label": "Fixed Deposit vs Recurring Deposit",  "a": "fixed_deposit",     "b": "recurring_deposit"},
    {"label": "Home Loan vs Personal Loan",           "a": "home_loan",         "b": "personal_loan"},
    {"label": "Home Loan vs Vehicle Loan",            "a": "home_loan",         "b": "vehicle_loan"},
    {"label": "Personal Loan vs Vehicle Loan",        "a": "personal_loan",     "b": "vehicle_loan"},
    {"label": "Savings Account vs Current Account",   "a": "savings_account",   "b": "current_account"},
    {"label": "Savings Account vs Salary Account",    "a": "savings_account",   "b": "salary_account"},
    {"label": "Credit Card vs Debit Card",            "a": "credit_card",       "b": "debit_card"},
]


# ── LangGraph state ────────────────────────────────────────────────────────────

class ComparisonState(TypedDict):
    product_a: str
    product_b: str
    context_a: str
    context_b: str
    parameters: list   # [{parameter, value_a, value_b, winner, note}]
    highlights: list   # [str]  — top 3-5 key differences as bullet points
    recommendation: str
    error: Optional[str]


# ── LLM singleton ─────────────────────────────────────────────────────────────

_llm: ChatGoogleGenerativeAI | None = None

def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash"),
            temperature=0.1,
        )
    return _llm


# ── Node 1: retrieve_both ──────────────────────────────────────────────────────

def retrieve_both_node(state: ComparisonState) -> dict:
    pa, pb = state["product_a"], state["product_b"]
    cfg_a = COMPARE_PRODUCT_CONFIG.get(pa)
    cfg_b = COMPARE_PRODUCT_CONFIG.get(pb)

    if not cfg_a:
        return {"error": f"Unknown product '{pa}'."}
    if not cfg_b:
        return {"error": f"Unknown product '{pb}'."}

    print(f"[Compare] Retrieving policy for: {cfg_a['label']} vs {cfg_b['label']}")
    try:
        vs = load_vector_store()
        ret_a = get_retriever(vs, domain=cfg_a["domain"], k=8)
        ret_b = get_retriever(vs, domain=cfg_b["domain"], k=8)
        docs_a = ret_a.invoke(cfg_a["search_query"])
        docs_b = ret_b.invoke(cfg_b["search_query"])
        ctx_a = "\n\n---\n\n".join(d.page_content for d in docs_a)
        ctx_b = "\n\n---\n\n".join(d.page_content for d in docs_b)
        print(f"[Compare] Retrieved {len(docs_a)} chunks for A, {len(docs_b)} for B")
        return {"context_a": ctx_a, "context_b": ctx_b}
    except Exception as exc:
        return {"error": f"RAG retrieval failed: {exc}"}


# ── Node 2: extract_parameters ─────────────────────────────────────────────────

def extract_parameters_node(state: ComparisonState) -> dict:
    if state.get("error"):
        return {}

    pa, pb = state["product_a"], state["product_b"]
    label_a = COMPARE_PRODUCT_CONFIG[pa]["label"]
    label_b = COMPARE_PRODUCT_CONFIG[pb]["label"]

    prompt = f"""You are a FinTrust Bank senior product manager preparing a side-by-side policy comparison.

PRODUCT A: {label_a}
POLICY EXCERPT A:
{state["context_a"]}

PRODUCT B: {label_b}
POLICY EXCERPT B:
{state["context_b"]}

TASK:
Extract 10–16 comparable policy parameters for these two products (e.g. interest rate, minimum amount, tenure, fees, eligibility, penalty, etc.).

For each parameter:
- "parameter": short name (e.g. "Interest Rate", "Minimum Amount", "Processing Fee")
- "value_a": extracted value for {label_a} — use "Not mentioned" if absent
- "value_b": extracted value for {label_b} — use "Not mentioned" if absent
- "winner": which product is more favourable on this parameter for the customer — must be one of: "{label_a}", "{label_b}", "Neutral", "Depends"
- "note": one-line explanation of why or any nuance (max 15 words)

Also provide:
- "highlights": list of 4–6 sentences summarising the most important differences (each ~20 words)
- "recommendation": 2–3 sentence guidance on which product suits which customer profile

Respond with ONLY valid JSON (no markdown fences):
{{
  "parameters": [{{"parameter":"...","value_a":"...","value_b":"...","winner":"...","note":"..."}}],
  "highlights": ["..."],
  "recommendation": "..."
}}"""

    print("[Compare] Extracting comparison parameters with LLM...")
    raw = ""
    try:
        raw = _get_llm().invoke(prompt).content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        print(f"[Compare] Extracted {len(data.get('parameters', []))} parameters")
        return {
            "parameters": data.get("parameters", []),
            "highlights": data.get("highlights", []),
            "recommendation": data.get("recommendation", ""),
        }
    except Exception as exc:
        return {"error": f"LLM extraction failed: {exc}. Raw: {raw[:300]}"}


# ── Node 3: build_comparison ───────────────────────────────────────────────────

def build_comparison_node(state: ComparisonState) -> dict:
    if state.get("error"):
        return {}
    # Nothing further to compute — state is already complete.
    return {}


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_comparison_graph():
    g = StateGraph(ComparisonState)
    g.add_node("retrieve_both",        retrieve_both_node)
    g.add_node("extract_parameters",   extract_parameters_node)
    g.add_node("build_comparison",     build_comparison_node)
    g.add_edge(START,                  "retrieve_both")
    g.add_edge("retrieve_both",        "extract_parameters")
    g.add_edge("extract_parameters",   "build_comparison")
    g.add_edge("build_comparison",     END)
    return g.compile()


_comparison_graph = None


def compare_products(product_a: str, product_b: str) -> dict:
    """
    Public API: compare two products side-by-side.
    Returns: product_a, product_b, label_a, label_b,
             parameters, highlights, recommendation, error
    """
    global _comparison_graph
    if _comparison_graph is None:
        _comparison_graph = build_comparison_graph()

    initial: ComparisonState = {
        "product_a": product_a,
        "product_b": product_b,
        "context_a": "",
        "context_b": "",
        "parameters": [],
        "highlights": [],
        "recommendation": "",
        "error": None,
    }
    result = _comparison_graph.invoke(initial)
    return {
        "product_a": product_a,
        "product_b": product_b,
        "label_a": COMPARE_PRODUCT_CONFIG.get(product_a, {}).get("label", product_a),
        "label_b": COMPARE_PRODUCT_CONFIG.get(product_b, {}).get("label", product_b),
        "parameters": result.get("parameters", []),
        "highlights": result.get("highlights", []),
        "recommendation": result.get("recommendation", ""),
        "error": result.get("error"),
    }


def list_comparable_products() -> list[dict]:
    """Return all products available for comparison."""
    return [
        {"product_id": k, "label": v["label"]}
        for k, v in COMPARE_PRODUCT_CONFIG.items()
    ]


def list_comparable_groups() -> list[dict]:
    """Return pre-defined quick-pick comparison pairs."""
    return COMPARABLE_GROUPS
