"""
agents/eligibility_agent.py

Loan & Account Eligibility Checker for FinTrust Compass.

LangGraph flow:
    [START] -> [extract_rules] -> [evaluate] -> [generate_report] -> [END]

Supported products:
    loans    : home_loan, personal_loan, vehicle_loan
    deposits : fixed_deposit, recurring_deposit
    accounts : savings_account, current_account, salary_account
    cards    : credit_card, debit_card
"""

from __future__ import annotations
import json, os, textwrap, warnings
from typing import Optional
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from ingestion.vector_store import get_retriever, load_vector_store

load_dotenv()

PRODUCT_CONFIG: dict[str, dict] = {
    "home_loan":        {"domain": "loans",    "label": "Home Loan",         "search_query": "home loan eligibility criteria age income employment CIBIL LTV maximum tenure"},
    "personal_loan":    {"domain": "loans",    "label": "Personal Loan",     "search_query": "personal loan eligibility criteria age income employment CIBIL score minimum salary"},
    "vehicle_loan":     {"domain": "loans",    "label": "Vehicle Loan",      "search_query": "vehicle loan eligibility criteria age income LTV margin down payment"},
    "fixed_deposit":    {"domain": "deposits", "label": "Fixed Deposit",     "search_query": "fixed deposit eligibility minimum amount tenure who can open"},
    "recurring_deposit":{"domain": "deposits", "label": "Recurring Deposit", "search_query": "recurring deposit eligibility minimum installment tenure criteria"},
    "savings_account":  {"domain": "accounts", "label": "Savings Account",   "search_query": "savings account eligibility KYC documents minimum balance who can open"},
    "current_account":  {"domain": "accounts", "label": "Current Account",   "search_query": "current account eligibility business documents KYC entity type"},
    "salary_account":   {"domain": "accounts", "label": "Salary Account",    "search_query": "salary account eligibility employer tie-up criteria salaried individual"},
    "credit_card":      {"domain": "cards",    "label": "Credit Card",       "search_query": "credit card eligibility age income CIBIL score employment criteria"},
    "debit_card":       {"domain": "cards",    "label": "Debit Card",        "search_query": "debit card eligibility linked savings account KYC criteria"},
}


class EligibilityState(TypedDict):
    product: str
    profile: dict
    policy_context: str
    criteria: list
    decision: str
    decision_reason: str
    conditions: list
    error: Optional[str]


_llm: ChatGoogleGenerativeAI | None = None

def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash"),
            temperature=0.1,
        )
    return _llm


def extract_rules_node(state: EligibilityState) -> dict:
    """RAG: retrieve eligibility policy text for the selected product."""
    product = state["product"]
    cfg = PRODUCT_CONFIG.get(product)
    if not cfg:
        return {"error": f"Unknown product '{product}'. Valid: {list(PRODUCT_CONFIG)}"}
    print(f"[Eligibility] Retrieving policy for: {cfg['label']}")
    try:
        vs = load_vector_store()
        retriever = get_retriever(vs, domain=cfg["domain"], k=8)
        docs = retriever.invoke(cfg["search_query"])
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        print(f"[Eligibility] Retrieved {len(docs)} chunks")
        return {"policy_context": context}
    except Exception as exc:
        return {"error": f"RAG retrieval failed: {exc}"}


def evaluate_node(state: EligibilityState) -> dict:
    """LLM evaluates each eligibility criterion against the applicant profile."""
    if state.get("error"):
        return {}
    product = state["product"]
    profile = state["profile"]
    context = state["policy_context"]
    label = PRODUCT_CONFIG[product]["label"]
    profile_text = "\n".join(f"  {k.replace('_',' ').title()}: {v}" for k, v in profile.items())

    prompt = (
        f"You are a senior banking officer at FinTrust Bank evaluating a customer's "
        f"eligibility for a **{label}**.\n\n"
        f"POLICY EXCERPT (from official FinTrust policy documents):\n{context}\n\n"
        f"APPLICANT PROFILE (these are ALL the fields available — nothing else is known):\n{profile_text}\n\n"
        f"STRICT RULES:\n"
        f"- Only include a criterion if EVERY piece of data needed to evaluate it is present "
        f"in the applicant profile above.\n"
        f"- If even one required data point for a criterion is absent from the profile, "
        f"**do not include that criterion at all**. Omit it silently.\n"
        f"- Never include a criterion just to say data is missing — skip it.\n"
        f"- For criteria you do include, assign:\n"
        f"   PASS — applicant clearly meets the requirement\n"
        f"   CONDITIONAL — applicant partially meets it and the provided data shows a borderline case\n"
        f"   FAIL — applicant clearly does not meet the requirement\n\n"
        f"Respond with ONLY valid JSON (no markdown fences):\n"
        f'{{"criteria": [{{"criterion": "...", "requirement": "...", '
        f'"applicant_value": "...", "status": "PASS|CONDITIONAL|FAIL", '
        f'"reason": "..."}}]}}\n\n'
        f"Minimum 3, maximum 10 criteria — only those fully evaluable from the profile above."
    )

    print("[Eligibility] Evaluating criteria with LLM...")
    raw = ""
    # Keywords in applicant_value or reason that signal the LLM couldn't
    # actually evaluate the criterion because data was absent from the profile.
    _MISSING_SIGNALS = (
        "not specified", "not provided", "not indicated", "not available",
        "not mentioned", "not in the profile", "not included", "absent",
        "no information", "no data", "unknown", "cannot be determined",
        "not confirmed", "not stated", "not given", "not present",
        "not supplied", "not furnished", "not submitted",
        # computation signals — criterion needs data not in profile
        "cannot be calculated", "cannot be computed", "not calculable",
        "is not available", "not possible to calculate", "not possible to assess",
        "insufficient data", "insufficient information",
        "does not specify", "does not indicate", "does not provide",
        "does not confirm", "does not include",
        "profile does not", "no profile data",
    )

    try:
        raw = _get_llm().invoke(prompt).content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        all_criteria = data["criteria"]

        # Drop any criterion where the LLM's own applicant_value or reason
        # signals that the data was simply not present in the profile.
        def _has_missing_signal(c: dict) -> bool:
            text = (c.get("applicant_value", "") + " " + c.get("reason", "")).lower()
            return any(sig in text for sig in _MISSING_SIGNALS)

        filtered = [c for c in all_criteria if not _has_missing_signal(c)]
        skipped  = len(all_criteria) - len(filtered)
        print(f"[Eligibility] Evaluated {len(all_criteria)} criteria, skipped {skipped} with missing data → {len(filtered)} shown")
        return {"criteria": filtered}
    except Exception as exc:
        return {"error": f"LLM evaluation failed: {exc}. Raw: {raw[:300]}"}


def generate_report_node(state: EligibilityState) -> dict:
    """Aggregate criterion scores into a final PASS / CONDITIONAL / FAIL decision."""
    if state.get("error"):
        return {"decision": "ERROR", "decision_reason": state["error"], "conditions": []}

    criteria = state.get("criteria", [])
    product  = state["product"]
    label    = PRODUCT_CONFIG[product]["label"]
    profile  = state["profile"]

    if not criteria:
        return {"decision": "FAIL", "decision_reason": "No criteria extracted from policy.", "conditions": []}

    fails        = [c for c in criteria if c["status"] == "FAIL"]
    conditionals = [c for c in criteria if c["status"] == "CONDITIONAL"]
    decision     = "FAIL" if fails else ("CONDITIONAL" if conditionals else "PASS")
    conditions   = [f"{c['criterion']}: {c['reason']}" for c in conditionals]

    criteria_lines = "\n".join(f"  [{c['status']:>12}] {c['criterion']}: {c['reason']}" for c in criteria)
    profile_text   = "\n".join(f"  {k.replace('_',' ').title()}: {v}" for k, v in profile.items())

    summary_prompt = (
        f"You are a senior banking officer at FinTrust Bank writing an internal note.\n\n"
        f"A customer applied for a **{label}** and received overall decision: **{decision}**.\n\n"
        f"Applicant profile:\n{profile_text}\n\n"
        f"Criterion results:\n{criteria_lines}\n\n"
        f"Write a clear, professional 2-3 sentence summary for the bank employee:\n"
        f"- State the decision and primary reason.\n"
        f"- If CONDITIONAL: what must the customer provide or clarify.\n"
        f"- If FAIL: which hard blockers disqualify them and what they can do next.\n"
        f"- If PASS: confirm they can proceed and state the next step.\n"
        f"Be factual, empathetic, and jargon-free."
    )

    print("[Eligibility] Generating decision summary...")
    try:
        decision_reason = _get_llm().invoke(summary_prompt).content.strip()
    except Exception:
        fails_str = "; ".join(c["criterion"] for c in fails) if fails else "none"
        cond_str  = "; ".join(c["criterion"] for c in conditionals) if conditionals else "none"
        decision_reason = f"Decision: {decision}. Blockers: {fails_str}. Conditional: {cond_str}."

    return {"decision": decision, "decision_reason": decision_reason, "conditions": conditions}


def build_eligibility_graph():
    g = StateGraph(EligibilityState)
    g.add_node("extract_rules",   extract_rules_node)
    g.add_node("evaluate",        evaluate_node)
    g.add_node("generate_report", generate_report_node)
    g.add_edge(START,             "extract_rules")
    g.add_edge("extract_rules",   "evaluate")
    g.add_edge("evaluate",        "generate_report")
    g.add_edge("generate_report", END)
    return g.compile()


_eligibility_graph = None

def check_eligibility(product: str, profile: dict) -> dict:
    """
    Public API: run the full eligibility check pipeline.
    Returns dict with: product, product_label, decision, decision_reason,
                       conditions, criteria, error
    """
    global _eligibility_graph
    if _eligibility_graph is None:
        _eligibility_graph = build_eligibility_graph()

    initial: EligibilityState = {
        "product": product, "profile": profile,
        "policy_context": "", "criteria": [],
        "decision": "", "decision_reason": "",
        "conditions": [], "error": None,
    }
    result = _eligibility_graph.invoke(initial)
    return {
        "product": product,
        "product_label": PRODUCT_CONFIG.get(product, {}).get("label", product),
        "decision": result.get("decision", "ERROR"),
        "decision_reason": result.get("decision_reason", ""),
        "conditions": result.get("conditions", []),
        "criteria": result.get("criteria", []),
        "error": result.get("error"),
    }


def list_products() -> list[dict]:
    return [{"product_id": k, "label": v["label"], "domain": v["domain"]} for k, v in PRODUCT_CONFIG.items()]
