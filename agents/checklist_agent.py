"""
agents/checklist_agent.py

Document Checklist Generator for FinTrust Compass.

LangGraph flow:
    [START] -> [retrieve_policy] -> [extract_checklist] -> [format_output] -> [END]

Given a product + applicant category, the agent:
  1. Retrieves the relevant document requirements from policy PDFs (RAG)
  2. Asks Gemini to extract a structured, categorised checklist
  3. Returns a clean list ready for the UI / export

Supported products: same set as eligibility_agent.PRODUCT_CONFIG
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

# ── product config ─────────────────────────────────────────────────────────────

PRODUCT_CONFIG: dict[str, dict] = {
    "home_loan":         {"domain": "loans",    "label": "Home Loan",         "search_query": "home loan documents required KYC identity address income property"},
    "personal_loan":     {"domain": "loans",    "label": "Personal Loan",     "search_query": "personal loan documents required KYC identity income salary employment"},
    "vehicle_loan":      {"domain": "loans",    "label": "Vehicle Loan",      "search_query": "vehicle loan documents required KYC identity income vehicle quotation"},
    "fixed_deposit":     {"domain": "deposits", "label": "Fixed Deposit",     "search_query": "fixed deposit documents required KYC identity address account opening"},
    "recurring_deposit": {"domain": "deposits", "label": "Recurring Deposit", "search_query": "recurring deposit documents required KYC identity address"},
    "savings_account":   {"domain": "accounts", "label": "Savings Account",   "search_query": "savings account documents required KYC identity address proof"},
    "current_account":   {"domain": "accounts", "label": "Current Account",   "search_query": "current account documents required KYC business entity proof"},
    "salary_account":    {"domain": "accounts", "label": "Salary Account",    "search_query": "salary account documents required employer letter identity proof"},
    "credit_card":       {"domain": "cards",    "label": "Credit Card",       "search_query": "credit card application documents required KYC income proof"},
    "debit_card":        {"domain": "cards",    "label": "Debit Card",        "search_query": "debit card documents required KYC account linked"},
}

APPLICANT_CATEGORIES = [
    "Salaried Individual",
    "Self-Employed Professional",
    "Self-Employed Business Owner",
    "Pensioner / Retired",
    "NRI (Non-Resident Indian)",
    "Minor (Guardian-operated)",
    "HUF (Hindu Undivided Family)",
    "Proprietorship",
    "Partnership Firm",
    "Private Limited Company",
    "LLP",
    "Trust / NGO",
]


# ── LangGraph state ────────────────────────────────────────────────────────────

class ChecklistState(TypedDict):
    product: str
    applicant_category: str
    additional_context: str          # optional extra info (e.g. "under-construction property")
    policy_context: str
    checklist: list                  # [{category, document, mandatory, notes, alternatives}]
    summary: str
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


# ── Node 1: retrieve_policy ────────────────────────────────────────────────────

def retrieve_policy_node(state: ChecklistState) -> dict:
    product = state["product"]
    cfg = PRODUCT_CONFIG.get(product)
    if not cfg:
        return {"error": f"Unknown product '{product}'."}
    print(f"[Checklist] Retrieving docs policy for: {cfg['label']}")
    try:
        vs = load_vector_store()
        retriever = get_retriever(vs, domain=cfg["domain"], k=8)
        docs = retriever.invoke(cfg["search_query"])
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        print(f"[Checklist] Retrieved {len(docs)} policy chunks")
        return {"policy_context": context}
    except Exception as exc:
        return {"error": f"RAG retrieval failed: {exc}"}


# ── Node 2: extract_checklist ──────────────────────────────────────────────────

def extract_checklist_node(state: ChecklistState) -> dict:
    if state.get("error"):
        return {}

    product   = state["product"]
    label     = PRODUCT_CONFIG[product]["label"]
    category  = state["applicant_category"]
    context   = state["policy_context"]
    extra     = state.get("additional_context", "").strip()
    extra_txt = f"\nADDITIONAL CONTEXT PROVIDED BY BANK EMPLOYEE:\n{extra}" if extra else ""

    prompt = (
        f"You are a senior banking officer at FinTrust Bank preparing a document "
        f"checklist for a **{label}** application.\n\n"
        f"APPLICANT CATEGORY: {category}\n"
        f"{extra_txt}\n\n"
        f"POLICY EXCERPT (from official FinTrust policy documents):\n{context}\n\n"
        f"TASK:\n"
        f"Extract a complete, structured document checklist from the policy for this "
        f"product and applicant category. Group documents into logical categories "
        f"(e.g. Identity Proof, Address Proof, Income Documents, Property Documents, "
        f"Business Documents, etc.).\n\n"
        f"For each document:\n"
        f"- mandatory: true if the policy explicitly requires it, false if optional\n"
        f"- alternatives: list other acceptable documents if the policy offers choices "
        f"  (e.g. Aadhaar OR Passport OR Voter ID), otherwise empty list\n"
        f"- notes: any policy-specific note (e.g. 'must be attested', 'last 3 months', "
        f"  'original + 2 copies') — keep it short\n\n"
        f"Respond with ONLY valid JSON (no markdown fences):\n"
        f'{{"checklist": [{{"category": "...", "document": "...", "mandatory": true|false, "alternatives": ["..."], "notes": "..."}}]}}\n\n'
        f"Include every document mentioned in the policy for this product+category combination. "
        f"Do not invent documents not mentioned in the policy."
    )

    print("[Checklist] Extracting checklist with LLM...")
    raw = ""
    try:
        raw = _get_llm().invoke(prompt).content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        items = data["checklist"]
        print(f"[Checklist] Extracted {len(items)} documents")
        return {"checklist": items}
    except Exception as exc:
        return {"error": f"LLM extraction failed: {exc}. Raw: {raw[:300]}"}


# ── Node 3: format_output ──────────────────────────────────────────────────────

def format_output_node(state: ChecklistState) -> dict:
    if state.get("error"):
        return {"summary": f"Error: {state['error']}"}

    checklist = state.get("checklist", [])
    product   = state["product"]
    label     = PRODUCT_CONFIG[product]["label"]
    category  = state["applicant_category"]

    if not checklist:
        return {"summary": "No documents could be extracted from the policy."}

    mandatory_count = sum(1 for c in checklist if c.get("mandatory", True))
    optional_count  = len(checklist) - mandatory_count

    # Group by category for the summary
    categories = {}
    for item in checklist:
        cat = item.get("category", "General")
        categories.setdefault(cat, []).append(item)

    cat_summary = ", ".join(f"{cat} ({len(items)})" for cat, items in categories.items())
    summary = (
        f"{label} — {category}: {len(checklist)} documents required "
        f"({mandatory_count} mandatory, {optional_count} optional). "
        f"Categories: {cat_summary}."
    )
    return {"summary": summary}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_checklist_graph():
    g = StateGraph(ChecklistState)
    g.add_node("retrieve_policy",   retrieve_policy_node)
    g.add_node("extract_checklist", extract_checklist_node)
    g.add_node("format_output",     format_output_node)
    g.add_edge(START,               "retrieve_policy")
    g.add_edge("retrieve_policy",   "extract_checklist")
    g.add_edge("extract_checklist", "format_output")
    g.add_edge("format_output",     END)
    return g.compile()


_checklist_graph = None

def generate_checklist(product: str, applicant_category: str, additional_context: str = "") -> dict:
    """
    Public API: generate a document checklist.
    Returns dict with: product, product_label, applicant_category,
                       checklist, summary, error
    """
    global _checklist_graph
    if _checklist_graph is None:
        _checklist_graph = build_checklist_graph()

    initial: ChecklistState = {
        "product": product,
        "applicant_category": applicant_category,
        "additional_context": additional_context,
        "policy_context": "",
        "checklist": [],
        "summary": "",
        "error": None,
    }
    result = _checklist_graph.invoke(initial)
    return {
        "product": product,
        "product_label": PRODUCT_CONFIG.get(product, {}).get("label", product),
        "applicant_category": applicant_category,
        "checklist": result.get("checklist", []),
        "summary": result.get("summary", ""),
        "error": result.get("error"),
    }


def list_applicant_categories() -> list[str]:
    return APPLICANT_CATEGORIES
