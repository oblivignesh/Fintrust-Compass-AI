"""
agents/fee_calculator_agent.py

Fee and Interest Calculator for FinTrust Compass.

LangGraph flow:
    [START] -> [retrieve_rate_policy] -> [compute_and_explain] -> [format_output] -> [END]

Given a product + financial parameters, the agent:
  1. Retrieves current fee/rate/charge details from policy PDFs (RAG)
  2. Performs exact arithmetic computations (EMI, interest, charges)
  3. Returns a structured breakdown with per-period schedules

Supported products: home_loan, personal_loan, vehicle_loan,
                    fixed_deposit, recurring_deposit, credit_card
"""

from __future__ import annotations
import json, math, os, warnings
from typing import Optional
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict
from ingestion.vector_store import get_retriever, load_vector_store

load_dotenv()

# ── Product catalogue ─────────────────────────────────────────────────────────

CALC_PRODUCT_CONFIG: dict[str, dict] = {
    "home_loan": {
        "label": "Home Loan",
        "domain": "loans",
        "calc_types": ["EMI", "Total Interest", "Amortisation Schedule"],
        "search_query": "home loan interest rate processing fee prepayment charges",
        "inputs": [
            {"key": "principal",      "label": "Loan Amount (₹)",          "type": "number", "default": 5000000},
            {"key": "annual_rate",    "label": "Annual Interest Rate (%)",  "type": "number", "default": 8.5},
            {"key": "tenure_months",  "label": "Tenure (months)",           "type": "number", "default": 240},
            {"key": "processing_fee_pct", "label": "Processing Fee (%)", "type": "number", "default": 0.5},
        ],
    },
    "personal_loan": {
        "label": "Personal Loan",
        "domain": "loans",
        "calc_types": ["EMI", "Total Interest", "Amortisation Schedule"],
        "search_query": "personal loan interest rate processing fee prepayment foreclosure charges",
        "inputs": [
            {"key": "principal",      "label": "Loan Amount (₹)",          "type": "number", "default": 500000},
            {"key": "annual_rate",    "label": "Annual Interest Rate (%)",  "type": "number", "default": 12.0},
            {"key": "tenure_months",  "label": "Tenure (months)",           "type": "number", "default": 60},
            {"key": "processing_fee_pct", "label": "Processing Fee (%)", "type": "number", "default": 2.0},
        ],
    },
    "vehicle_loan": {
        "label": "Vehicle Loan",
        "domain": "loans",
        "calc_types": ["EMI", "Total Interest", "Amortisation Schedule"],
        "search_query": "vehicle loan interest rate processing fee prepayment charges",
        "inputs": [
            {"key": "principal",      "label": "Loan Amount (₹)",          "type": "number", "default": 800000},
            {"key": "annual_rate",    "label": "Annual Interest Rate (%)",  "type": "number", "default": 9.5},
            {"key": "tenure_months",  "label": "Tenure (months)",           "type": "number", "default": 84},
            {"key": "processing_fee_pct", "label": "Processing Fee (%)", "type": "number", "default": 1.0},
        ],
    },
    "fixed_deposit": {
        "label": "Fixed Deposit",
        "domain": "deposits",
        "calc_types": ["Maturity Amount", "Interest Earned", "Quarterly Payout"],
        "search_query": "fixed deposit interest rate maturity calculation compounding TDS",
        "inputs": [
            {"key": "principal",      "label": "Deposit Amount (₹)",       "type": "number", "default": 100000},
            {"key": "annual_rate",    "label": "Annual Interest Rate (%)",  "type": "number", "default": 7.0},
            {"key": "tenure_months",  "label": "Tenure (months)",           "type": "number", "default": 12},
            {"key": "compounding",    "label": "Compounding Frequency",     "type": "select",
             "options": ["Monthly", "Quarterly", "Half-yearly", "Annually"], "default": "Quarterly"},
        ],
    },
    "recurring_deposit": {
        "label": "Recurring Deposit",
        "domain": "deposits",
        "calc_types": ["Maturity Amount", "Total Interest Earned"],
        "search_query": "recurring deposit interest rate maturity calculation monthly installment",
        "inputs": [
            {"key": "monthly_installment", "label": "Monthly Installment (₹)", "type": "number", "default": 5000},
            {"key": "annual_rate",         "label": "Annual Interest Rate (%)", "type": "number", "default": 6.5},
            {"key": "tenure_months",       "label": "Tenure (months)",          "type": "number", "default": 24},
        ],
    },
    "credit_card": {
        "label": "Credit Card",
        "domain": "cards",
        "calc_types": ["Interest on Outstanding", "Minimum Due", "Late Payment Fee"],
        "search_query": "credit card interest rate outstanding balance minimum due late payment fee annual fee",
        "inputs": [
            {"key": "outstanding_balance", "label": "Outstanding Balance (₹)",   "type": "number", "default": 50000},
            {"key": "annual_rate",         "label": "Annual Interest Rate (%)",   "type": "number", "default": 42.0},
            {"key": "min_due_pct",         "label": "Minimum Due (%)",            "type": "number", "default": 5.0},
            {"key": "days_overdue",        "label": "Days Since Due Date",        "type": "number", "default": 0},
        ],
    },
}


# ── LangGraph state ────────────────────────────────────────────────────────────

class FeeCalcState(TypedDict):
    product: str
    params: dict            # raw input parameters
    policy_context: str
    results: dict           # computed values
    schedule: list          # amortisation / period-wise breakdown (up to 12 rows)
    policy_notes: list      # notable fee/charge sentences from policy
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
            request_timeout=240,
        )
    return _llm


# ── Pure arithmetic helpers ───────────────────────────────────────────────

def _emi(principal: float, annual_rate: float, tenure_months: int) -> float:
    """Standard reducing-balance EMI formula."""
    if annual_rate == 0:
        return principal / tenure_months
    r = annual_rate / 12 / 100
    return principal * r * (1 + r) ** tenure_months / ((1 + r) ** tenure_months - 1)


def _amortisation(principal: float, annual_rate: float, tenure_months: int) -> list[dict]:
    """Return month-by-month amortisation schedule."""
    r = annual_rate / 12 / 100
    emi = _emi(principal, annual_rate, tenure_months)
    balance = principal
    schedule = []
    for m in range(1, tenure_months + 1):
        interest = balance * r
        principal_part = emi - interest
        balance = max(0.0, balance - principal_part)
        schedule.append({
            "month": m,
            "emi": round(emi, 2),
            "principal": round(principal_part, 2),
            "interest": round(interest, 2),
            "balance": round(balance, 2),
        })
    return schedule


def _fd_maturity(principal: float, annual_rate: float, tenure_months: int, compounding: str) -> float:
    freq_map = {"Monthly": 12, "Quarterly": 4, "Half-yearly": 2, "Annually": 1}
    n = freq_map.get(compounding, 4)
    r = annual_rate / 100
    t = tenure_months / 12
    return principal * (1 + r / n) ** (n * t)


def _rd_maturity(monthly: float, annual_rate: float, tenure_months: int) -> float:
    """RD maturity using quarterly compounding (standard Indian bank formula)."""
    r = annual_rate / 400   # quarterly rate
    total = 0.0
    for i in range(tenure_months):
        # remaining quarters after installment i
        quarters = (tenure_months - i) / 3
        total += monthly * (1 + r) ** quarters
    return total


def _compute_results(product: str, params: dict) -> tuple[dict, list]:
    """Perform all arithmetic. Returns (results_dict, schedule_list)."""
    results: dict = {}
    schedule: list = []

    p = params

    if product in ("home_loan", "personal_loan", "vehicle_loan"):
        principal      = float(p.get("principal", 0))
        annual_rate    = float(p.get("annual_rate", 0))
        tenure_months  = int(p.get("tenure_months", 0))
        fee_pct        = float(p.get("processing_fee_pct", 0))

        emi            = _emi(principal, annual_rate, tenure_months)
        total_payment  = emi * tenure_months
        total_interest = total_payment - principal
        processing_fee = principal * fee_pct / 100

        results = {
            "EMI (₹)":                f"{emi:,.2f}",
            "Total Payment (₹)":      f"{total_payment:,.2f}",
            "Total Interest (₹)":     f"{total_interest:,.2f}",
            "Processing Fee (₹)":     f"{processing_fee:,.2f}",
            "Effective Cost (₹)":     f"{total_payment + processing_fee:,.2f}",
            "Interest / Principal %": f"{total_interest / principal * 100:.1f}%",
        }
        full_sched = _amortisation(principal, annual_rate, tenure_months)
        # Return first 6 + last 6 rows for display
        if len(full_sched) > 12:
            schedule = full_sched[:6] + [{"month": "...", "emi": "...", "principal": "...",
                                          "interest": "...", "balance": "..."}] + full_sched[-6:]
        else:
            schedule = full_sched

    elif product == "fixed_deposit":
        principal     = float(p.get("principal", 0))
        annual_rate   = float(p.get("annual_rate", 0))
        tenure_months = int(p.get("tenure_months", 0))
        compounding   = p.get("compounding", "Quarterly")

        maturity      = _fd_maturity(principal, annual_rate, tenure_months, compounding)
        interest      = maturity - principal

        # Quarterly payout (simple interest per quarter)
        quarterly_payout = principal * annual_rate / 100 / 4

        results = {
            "Maturity Amount (₹)":   f"{maturity:,.2f}",
            "Interest Earned (₹)":   f"{interest:,.2f}",
            "Effective Annual Yield": f"{(maturity / principal) ** (12 / tenure_months) - 1:.3%}",
            "Quarterly Payout (₹)":  f"{quarterly_payout:,.2f}",
        }
        # Year-wise schedule
        for yr in range(1, math.ceil(tenure_months / 12) + 1):
            t = min(yr * 12, tenure_months)
            amt = _fd_maturity(principal, annual_rate, t, compounding)
            schedule.append({"year": yr, "value": round(amt, 2), "interest": round(amt - principal, 2)})

    elif product == "recurring_deposit":
        monthly       = float(p.get("monthly_installment", 0))
        annual_rate   = float(p.get("annual_rate", 0))
        tenure_months = int(p.get("tenure_months", 0))

        maturity      = _rd_maturity(monthly, annual_rate, tenure_months)
        invested      = monthly * tenure_months
        interest      = maturity - invested

        results = {
            "Total Invested (₹)":   f"{invested:,.2f}",
            "Maturity Amount (₹)":  f"{maturity:,.2f}",
            "Interest Earned (₹)":  f"{interest:,.2f}",
            "Effective Return %":   f"{interest / invested * 100:.2f}%",
        }
        for m in range(1, tenure_months + 1, max(1, tenure_months // 12)):
            amt = _rd_maturity(monthly, annual_rate, m)
            schedule.append({"month": m, "value": round(amt, 2), "invested": round(monthly * m, 2)})

    elif product == "credit_card":
        balance       = float(p.get("outstanding_balance", 0))
        annual_rate   = float(p.get("annual_rate", 42.0))
        min_due_pct   = float(p.get("min_due_pct", 5.0))
        days_overdue  = int(p.get("days_overdue", 0))

        monthly_rate  = annual_rate / 12 / 100
        daily_rate    = annual_rate / 365 / 100
        monthly_int   = balance * monthly_rate
        min_due       = max(balance * min_due_pct / 100, 200)

        # Late payment fee tiers (indicative)
        if days_overdue == 0:
            late_fee = 0
        elif balance < 500:
            late_fee = 0
        elif balance < 10000:
            late_fee = 500
        elif balance < 25000:
            late_fee = 750
        elif balance < 50000:
            late_fee = 1000
        elif balance < 100000:
            late_fee = 1200
        else:
            late_fee = 1300

        results = {
            "Monthly Interest (₹)":   f"{monthly_int:,.2f}",
            "Daily Interest (₹)":     f"{balance * daily_rate:,.2f}",
            "Minimum Due (₹)":        f"{min_due:,.2f}",
            "Late Payment Fee (₹)":   f"{late_fee:,.2f}" if days_overdue > 0 else "N/A",
            "Total Due if Overdue (₹)": f"{balance + monthly_int + late_fee:,.2f}" if days_overdue > 0 else "N/A",
        }

    return results, schedule


# ── Node 1: retrieve_rate_policy ───────────────────────────────────────────────

def retrieve_rate_policy_node(state: FeeCalcState) -> dict:
    product = state["product"]
    cfg = CALC_PRODUCT_CONFIG.get(product)
    if not cfg:
        return {"error": f"Unknown product '{product}'."}
    print(f"[FeeCalc] Retrieving rate/fee policy for: {cfg['label']}")
    try:
        vs = load_vector_store()
        retriever = get_retriever(vs, domain=cfg["domain"], k=8)
        docs = retriever.invoke(cfg["search_query"])
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        print(f"[FeeCalc] Retrieved {len(docs)} policy chunks")
        return {"policy_context": context}
    except Exception as exc:
        return {"error": f"RAG retrieval failed: {exc}"}


# ── Node 2: compute_and_explain ────────────────────────────────────────────────

def compute_and_explain_node(state: FeeCalcState) -> dict:
    if state.get("error"):
        return {}

    product = state["product"]
    params  = state["params"]
    cfg     = CALC_PRODUCT_CONFIG[product]

    # Step A: pure arithmetic (no LLM needed)
    try:
        results, schedule = _compute_results(product, params)
    except Exception as exc:
        return {"error": f"Computation error: {exc}"}

    # Step B: LLM extracts notable policy notes (fees, conditions, penalties)
    prompt = (
        f"You are a FinTrust Bank officer reviewing the fee/rate policy for "
        f"**{cfg['label']}**.\n\n"
        f"POLICY EXCERPT:\n{state['policy_context']}\n\n"
        f"TASK: Extract up to 8 short, factual sentences about fees, interest rates, "
        f"charges, penalties, or special conditions mentioned in the policy that are "
        f"relevant to a customer taking this product. "
        f"Each sentence must start with a relevant emoji (💰 for fees, 📊 for rates, "
        f"⚠️ for penalties, ℹ️ for general conditions).\n\n"
        f"Respond with ONLY valid JSON (no markdown fences):\n"
        f'{{"notes": ["...", "..."]}}\n\n'
        f"If no specific notes are found, return {{'notes': []}}."
    )

    print("[FeeCalc] Extracting policy notes with LLM...")
    raw = ""
    try:
        raw = _get_llm().invoke(prompt).content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        policy_notes = data.get("notes", [])
        print(f"[FeeCalc] Extracted {len(policy_notes)} policy notes")
    except Exception:
        policy_notes = []

    return {"results": results, "schedule": schedule, "policy_notes": policy_notes}


# ── Node 3: format_output ──────────────────────────────────────────────────────

def format_output_node(state: FeeCalcState) -> dict:
    if state.get("error"):
        return {"summary": f"Error: {state['error']}"}

    product = state["product"]
    label   = CALC_PRODUCT_CONFIG[product]["label"]
    results = state.get("results", {})

    if not results:
        return {"summary": "No results computed."}

    # Build human-readable summary from first few result lines
    top = list(results.items())[:3]
    top_str = "  |  ".join(f"{k}: {v}" for k, v in top)
    summary = f"{label}: {top_str}"
    return {"summary": summary}


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_fee_calc_graph():
    g = StateGraph(FeeCalcState)
    g.add_node("retrieve_rate_policy",  retrieve_rate_policy_node)
    g.add_node("compute_and_explain",   compute_and_explain_node)
    g.add_node("format_output",         format_output_node)
    g.add_edge(START,                    "retrieve_rate_policy")
    g.add_edge("retrieve_rate_policy",   "compute_and_explain")
    g.add_edge("compute_and_explain",    "format_output")
    g.add_edge("format_output",          END)
    return g.compile()


_fee_calc_graph = None


def calculate_fees(product: str, params: dict) -> dict:
    """
    Public API: compute fees/interest + retrieve policy notes.
    Returns: product, product_label, results, schedule, policy_notes, summary, error
    """
    global _fee_calc_graph
    if _fee_calc_graph is None:
        _fee_calc_graph = build_fee_calc_graph()

    initial: FeeCalcState = {
        "product": product,
        "params": params,
        "policy_context": "",
        "results": {},
        "schedule": [],
        "policy_notes": [],
        "summary": "",
        "error": None,
    }
    result = _fee_calc_graph.invoke(initial)
    return {
        "product": product,
        "product_label": CALC_PRODUCT_CONFIG.get(product, {}).get("label", product),
        "results": result.get("results", {}),
        "schedule": result.get("schedule", []),
        "policy_notes": result.get("policy_notes", []),
        "summary": result.get("summary", ""),
        "error": result.get("error"),
    }


def list_calc_products() -> list[dict]:
    return [
        {
            "product_id": k,
            "label": v["label"],
            "calc_types": v["calc_types"],
            "inputs": v["inputs"],
        }
        for k, v in CALC_PRODUCT_CONFIG.items()
    ]
