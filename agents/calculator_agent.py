"""
agents/calculator_agent.py

Fee and Interest Calculator for FinTrust Compass.

LangGraph flow:
    [START] -> [retrieve_rates] -> [extract_rates] -> [compute] -> [format_output] -> [END]

Given a product + user-supplied numeric inputs, the agent:
  1. Retrieves the applicable rate / fee schedule from policy PDFs (RAG)
  2. Asks Gemini to extract the specific rates as structured JSON
  3. Runs all maths in Python (no LLM for arithmetic)
  4. Returns a full breakdown ready for the UI

Supported calculations
──────────────────────
  loan_emi     – EMI, total interest, processing fee  (home / personal / vehicle loans)
  fd_maturity  – Maturity amount, interest earned     (fixed deposit)
  rd_maturity  – Maturity amount, interest earned     (recurring deposit)
  savings_int  – Quarterly / annual interest           (savings account)
  card_int     – Monthly interest charge, annual fee   (credit card)
"""

from __future__ import annotations
import json
import math
import os
import warnings
from typing import Optional

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from ingestion.vector_store import get_retriever, load_vector_store

load_dotenv()

# ── product / calculation config ───────────────────────────────────────────────

CALC_CONFIG: dict[str, dict] = {
    "home_loan": {
        "label":      "Home Loan",
        "domain":     "loans",
        "calc_type":  "loan_emi",
        "search_query": "home loan interest rate processing fee prepayment penalty charges schedule",
        "fallback_rate": 8.50,          # % p.a.
        "fallback_processing_fee_pct": 0.50,
        "fallback_processing_fee_max": 15_000,
        "fallback_prepayment_pct": 2.0,
    },
    "personal_loan": {
        "label":      "Personal Loan",
        "domain":     "loans",
        "calc_type":  "loan_emi",
        "search_query": "personal loan interest rate processing fee prepayment charges",
        "fallback_rate": 12.00,
        "fallback_processing_fee_pct": 1.00,
        "fallback_processing_fee_max": 10_000,
        "fallback_prepayment_pct": 3.0,
    },
    "vehicle_loan": {
        "label":      "Vehicle Loan",
        "domain":     "loans",
        "calc_type":  "loan_emi",
        "search_query": "vehicle loan auto loan interest rate processing fee prepayment charges",
        "fallback_rate": 9.50,
        "fallback_processing_fee_pct": 0.50,
        "fallback_processing_fee_max": 8_000,
        "fallback_prepayment_pct": 2.0,
    },
    "fixed_deposit": {
        "label":      "Fixed Deposit",
        "domain":     "deposits",
        "calc_type":  "fd_maturity",
        "search_query": "fixed deposit interest rate compounding maturity TDS premature withdrawal penalty",
        "fallback_rate": 6.50,
        "fallback_compounding_freq": 4,   # quarterly
        "fallback_tds_rate": 10.0,
        "fallback_premature_penalty": 1.0,
    },
    "recurring_deposit": {
        "label":      "Recurring Deposit",
        "domain":     "deposits",
        "calc_type":  "rd_maturity",
        "search_query": "recurring deposit interest rate compounding maturity premature withdrawal",
        "fallback_rate": 6.50,
        "fallback_compounding_freq": 4,
    },
    "savings_account": {
        "label":      "Savings Account",
        "domain":     "accounts",
        "calc_type":  "savings_int",
        "search_query": "savings account interest rate calculation balance quarterly compounding",
        "fallback_rate": 3.50,
    },
    "credit_card": {
        "label":      "Credit Card",
        "domain":     "cards",
        "calc_type":  "card_int",
        "search_query": "credit card interest rate finance charge monthly annual fee late payment penalty",
        "fallback_monthly_rate": 3.50,   # % per month
        "fallback_annual_fee": 500,
        "fallback_late_payment_fee": 750,
    },
}

# Input schema per calc_type — used by the API + frontend to know what fields to show
CALC_INPUT_SCHEMA: dict[str, list[dict]] = {
    "loan_emi": [
        {"key": "loan_amount",  "label": "Loan Amount (₹)",   "min": 10_000,    "max": 100_000_000, "default": 2_000_000,  "step": 10_000},
        {"key": "tenure_years", "label": "Tenure (Years)",    "min": 1,         "max": 30,          "default": 20,         "step": 1},
    ],
    "fd_maturity": [
        {"key": "principal",       "label": "Deposit Amount (₹)",  "min": 1_000,  "max": 50_000_000, "default": 100_000,  "step": 1_000},
        {"key": "tenure_months",   "label": "Tenure (Months)",     "min": 1,      "max": 120,        "default": 12,       "step": 1},
    ],
    "rd_maturity": [
        {"key": "monthly_installment", "label": "Monthly Installment (₹)", "min": 100,   "max": 1_000_000, "default": 5_000,  "step": 100},
        {"key": "tenure_months",       "label": "Tenure (Months)",         "min": 3,     "max": 120,       "default": 24,     "step": 1},
    ],
    "savings_int": [
        {"key": "average_balance", "label": "Average Quarterly Balance (₹)", "min": 0, "max": 50_000_000, "default": 50_000, "step": 1_000},
    ],
    "card_int": [
        {"key": "outstanding_balance", "label": "Outstanding Balance (₹)", "min": 0, "max": 5_000_000, "default": 25_000, "step": 500},
    ],
}


def list_calculator_products() -> list[dict]:
    return [
        {
            "product_id":  pid,
            "label":       cfg["label"],
            "calc_type":   cfg["calc_type"],
            "input_schema": CALC_INPUT_SCHEMA[cfg["calc_type"]],
        }
        for pid, cfg in CALC_CONFIG.items()
    ]


# ── LangGraph state ────────────────────────────────────────────────────────────

class CalculatorState(TypedDict):
    product: str
    inputs: dict                  # user-supplied numbers
    policy_context: str
    rates: dict                   # extracted from policy
    results: list                 # [{label, value, is_key, note}]
    assumptions: list             # strings describing what defaults were used
    summary: str
    error: Optional[str]


# ── LLM singleton ─────────────────────────────────────────────────────────────

_llm: ChatGoogleGenerativeAI | None = None

def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash"),
            temperature=0.0,
        )
    return _llm


# ── Node 1: retrieve_rates ─────────────────────────────────────────────────────

def retrieve_rates_node(state: CalculatorState) -> dict:
    product = state["product"]
    cfg = CALC_CONFIG.get(product)
    if not cfg:
        return {"error": f"Unknown product '{product}'."}
    print(f"[Calculator] Retrieving rate policy for: {cfg['label']}")
    try:
        vs = load_vector_store()
        retriever = get_retriever(vs, domain=cfg["domain"], k=8)
        docs = retriever.invoke(cfg["search_query"])
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        print(f"[Calculator] Retrieved {len(docs)} policy chunks")
        return {"policy_context": context}
    except Exception as exc:
        return {"error": f"RAG retrieval failed: {exc}"}


# ── Node 2: extract_rates ──────────────────────────────────────────────────────

_RATE_PROMPTS: dict[str, str] = {
    "loan_emi": (
        'Extract the following from the policy. Return ONLY valid JSON (no markdown):\n'
        '{"interest_rate_pa": <float>, "processing_fee_pct": <float>, '
        '"processing_fee_max_amount": <float or null>, '
        '"prepayment_charges_pct": <float>, "notes": ["..."]}\n\n'
        'If a value is not mentioned, use null. Do not guess.'
    ),
    "fd_maturity": (
        'Extract the following from the policy. Return ONLY valid JSON (no markdown):\n'
        '{"interest_rate_pa": <float>, "compounding_frequency_per_year": <int 1|4|12>, '
        '"tds_rate_pct": <float or null>, '
        '"premature_withdrawal_penalty_pct": <float or null>, "notes": ["..."]}\n\n'
        'compounding_frequency_per_year: 1=annual, 4=quarterly, 12=monthly. '
        'If a value is not mentioned, use null.'
    ),
    "rd_maturity": (
        'Extract the following from the policy. Return ONLY valid JSON (no markdown):\n'
        '{"interest_rate_pa": <float>, "compounding_frequency_per_year": <int 1|4|12>, '
        '"notes": ["..."]}\n\n'
        'If a value is not mentioned, use null.'
    ),
    "savings_int": (
        'Extract the following from the policy. Return ONLY valid JSON (no markdown):\n'
        '{"interest_rate_pa": <float>, "compounding_frequency": <"quarterly"|"monthly"|"annual">, '
        '"minimum_balance": <float or null>, "notes": ["..."]}\n\n'
        'If a value is not mentioned, use null.'
    ),
    "card_int": (
        'Extract the following from the policy. Return ONLY valid JSON (no markdown):\n'
        '{"monthly_interest_rate_pct": <float>, "annual_fee": <float or null>, '
        '"late_payment_fee": <float or null>, "minimum_payment_pct": <float or null>, '
        '"notes": ["..."]}\n\n'
        'If a value is not mentioned, use null.'
    ),
}


def extract_rates_node(state: CalculatorState) -> dict:
    if state.get("error"):
        return {}

    product   = state["product"]
    cfg       = CALC_CONFIG[product]
    calc_type = cfg["calc_type"]
    context   = state["policy_context"]

    prompt = (
        f"You are a FinTrust Bank rate officer reading official policy documents for "
        f"**{cfg['label']}**.\n\n"
        f"POLICY EXCERPT:\n{context}\n\n"
        f"{_RATE_PROMPTS[calc_type]}"
    )

    print("[Calculator] Extracting rates with LLM...")
    raw = ""
    try:
        raw = _get_llm().invoke(prompt).content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        rates = json.loads(raw)
        print(f"[Calculator] Extracted rates: {rates}")
        return {"rates": rates}
    except Exception as exc:
        print(f"[Calculator] Rate extraction failed ({exc}), using fallbacks")
        return {"rates": {}}


# ── Node 3: compute ────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    """Format rupee amount with commas and 2 decimals."""
    if v >= 1_00_00_000:
        return f"₹{v/1_00_00_000:.2f} Cr"
    if v >= 1_00_000:
        return f"₹{v/1_00_000:.2f} L"
    return f"₹{v:,.2f}"


def _resolve(rates: dict, key: str, fallback):
    """Return rates[key] if present and not None, else fallback."""
    val = rates.get(key)
    return val if val is not None else fallback


def _compute_loan_emi(inputs: dict, rates: dict, cfg: dict) -> tuple[list, list]:
    P  = float(inputs.get("loan_amount", 1_000_000))
    t  = float(inputs.get("tenure_years", 20))
    n  = int(t * 12)

    annual_rate    = _resolve(rates, "interest_rate_pa",            cfg["fallback_rate"])
    fee_pct        = _resolve(rates, "processing_fee_pct",          cfg["fallback_processing_fee_pct"])
    fee_max        = _resolve(rates, "processing_fee_max_amount",   cfg["fallback_processing_fee_max"])
    prepay_pct     = _resolve(rates, "prepayment_charges_pct",      cfg["fallback_prepayment_pct"])

    assumptions = []
    if rates.get("interest_rate_pa") is None:
        assumptions.append(f"Interest rate not found in policy — used indicative {annual_rate}% p.a.")
    if rates.get("processing_fee_pct") is None:
        assumptions.append(f"Processing fee not found in policy — used indicative {fee_pct}%.")

    r = annual_rate / 1200.0   # monthly rate as decimal
    if r == 0:
        emi = P / n
    else:
        emi = P * r * (1 + r) ** n / ((1 + r) ** n - 1)

    total_payable  = emi * n
    total_interest = total_payable - P
    proc_fee       = min(P * fee_pct / 100.0, fee_max) if fee_max else P * fee_pct / 100.0

    results = [
        {"label": "Monthly EMI",            "value": _fmt(emi),             "is_key": True,  "note": ""},
        {"label": "Loan Amount",            "value": _fmt(P),               "is_key": False, "note": ""},
        {"label": "Tenure",                 "value": f"{int(t)} years ({n} months)", "is_key": False, "note": ""},
        {"label": "Interest Rate",          "value": f"{annual_rate:.2f}% p.a.",     "is_key": False, "note": ""},
        {"label": "Total Interest Payable", "value": _fmt(total_interest),   "is_key": True,  "note": ""},
        {"label": "Total Amount Payable",   "value": _fmt(total_payable),    "is_key": False, "note": ""},
        {"label": "Processing Fee",         "value": _fmt(proc_fee),         "is_key": False, "note": f"{fee_pct}% of loan amount"},
        {"label": "Prepayment Charges",     "value": f"{prepay_pct:.2f}% of outstanding", "is_key": False, "note": "on foreclosure"},
    ]
    return results, assumptions


def _compute_fd_maturity(inputs: dict, rates: dict, cfg: dict) -> tuple[list, list]:
    P           = float(inputs.get("principal", 100_000))
    tm          = float(inputs.get("tenure_months", 12))
    t           = tm / 12.0

    rate        = _resolve(rates, "interest_rate_pa",               cfg["fallback_rate"])
    freq        = _resolve(rates, "compounding_frequency_per_year", cfg["fallback_compounding_freq"])
    tds_rate    = _resolve(rates, "tds_rate_pct",                   cfg.get("fallback_tds_rate", 10.0))
    penalty_pct = _resolve(rates, "premature_withdrawal_penalty_pct", cfg.get("fallback_premature_penalty", 1.0))

    freq = int(freq) if freq else 4
    assumptions = []
    if rates.get("interest_rate_pa") is None:
        assumptions.append(f"Interest rate not found in policy — used indicative {rate}% p.a.")
    if rates.get("compounding_frequency_per_year") is None:
        assumptions.append(f"Compounding frequency not found — assumed quarterly.")

    maturity       = P * (1 + rate / (100.0 * freq)) ** (freq * t)
    interest       = maturity - P
    tds_deducted   = interest * tds_rate / 100.0
    net_interest   = interest - tds_deducted
    net_maturity   = P + net_interest
    effective_rate = ((maturity / P) ** (1 / t) - 1) * 100 if t > 0 else 0

    freq_label     = {1: "Annual", 4: "Quarterly", 12: "Monthly"}.get(freq, f"{freq}x/year")
    premature_val  = P * penalty_pct / 100.0

    results = [
        {"label": "Maturity Amount (Gross)",     "value": _fmt(maturity),       "is_key": True,  "note": "before TDS"},
        {"label": "Maturity Amount (Net of TDS)","value": _fmt(net_maturity),   "is_key": True,  "note": f"after {tds_rate}% TDS on interest"},
        {"label": "Principal",                   "value": _fmt(P),              "is_key": False, "note": ""},
        {"label": "Tenure",                      "value": f"{tm:.0f} months ({t:.2f} years)", "is_key": False, "note": ""},
        {"label": "Interest Rate",               "value": f"{rate:.2f}% p.a.",  "is_key": False, "note": f"{freq_label} compounding"},
        {"label": "Effective Annual Yield",      "value": f"{effective_rate:.2f}% p.a.", "is_key": False, "note": ""},
        {"label": "Gross Interest Earned",       "value": _fmt(interest),       "is_key": False, "note": ""},
        {"label": "TDS Deducted",                "value": _fmt(tds_deducted),   "is_key": False, "note": f"{tds_rate}% on interest"},
        {"label": "Net Interest Earned",         "value": _fmt(net_interest),   "is_key": False, "note": ""},
        {"label": "Premature Withdrawal Penalty","value": _fmt(premature_val),  "is_key": False, "note": f"{penalty_pct}% of principal"},
    ]
    return results, assumptions


def _compute_rd_maturity(inputs: dict, rates: dict, cfg: dict) -> tuple[list, list]:
    P_monthly = float(inputs.get("monthly_installment", 5_000))
    tm        = int(inputs.get("tenure_months", 24))
    rate      = _resolve(rates, "interest_rate_pa",               cfg["fallback_rate"])
    freq      = _resolve(rates, "compounding_frequency_per_year", cfg["fallback_compounding_freq"])
    freq      = int(freq) if freq else 4

    assumptions = []
    if rates.get("interest_rate_pa") is None:
        assumptions.append(f"Interest rate not found in policy — used indicative {rate}% p.a.")

    # Exact: each monthly installment compounds for its remaining term
    monthly_rate = rate / 1200.0
    maturity = sum(P_monthly * (1 + monthly_rate) ** (tm - i) for i in range(tm))
    total_invested = P_monthly * tm
    interest = maturity - total_invested

    results = [
        {"label": "Maturity Amount",      "value": _fmt(maturity),        "is_key": True,  "note": ""},
        {"label": "Total Amount Invested","value": _fmt(total_invested),  "is_key": False, "note": f"₹{P_monthly:,.0f} × {tm} months"},
        {"label": "Interest Earned",      "value": _fmt(interest),        "is_key": True,  "note": ""},
        {"label": "Tenure",               "value": f"{tm} months",        "is_key": False, "note": ""},
        {"label": "Interest Rate",        "value": f"{rate:.2f}% p.a.",   "is_key": False, "note": "monthly compounding"},
        {"label": "Effective Yield",      "value": f"{(interest/total_invested*100):.2f}% total return", "is_key": False, "note": ""},
    ]
    return results, assumptions


def _compute_savings_int(inputs: dict, rates: dict, cfg: dict) -> tuple[list, list]:
    bal  = float(inputs.get("average_balance", 50_000))
    rate = _resolve(rates, "interest_rate_pa", cfg["fallback_rate"])

    assumptions = []
    if rates.get("interest_rate_pa") is None:
        assumptions.append(f"Interest rate not found in policy — used indicative {rate}% p.a.")

    # Quarterly compounding (standard for savings accounts in India)
    quarterly_rate    = rate / 400.0
    quarterly_int     = bal * quarterly_rate
    annual_int_simple = bal * rate / 100.0
    annual_int_compnd = bal * (1 + quarterly_rate) ** 4 - bal
    daily_int         = bal * rate / 36500.0

    results = [
        {"label": "Annual Interest (Compound)", "value": _fmt(annual_int_compnd), "is_key": True,  "note": "quarterly compounding"},
        {"label": "Annual Interest (Simple)",   "value": _fmt(annual_int_simple), "is_key": False, "note": ""},
        {"label": "Quarterly Interest",         "value": _fmt(quarterly_int),     "is_key": False, "note": ""},
        {"label": "Daily Interest (approx.)",   "value": _fmt(daily_int),         "is_key": False, "note": ""},
        {"label": "Average Balance",            "value": _fmt(bal),               "is_key": False, "note": ""},
        {"label": "Interest Rate",              "value": f"{rate:.2f}% p.a.",     "is_key": False, "note": ""},
    ]
    return results, assumptions


def _compute_card_int(inputs: dict, rates: dict, cfg: dict) -> tuple[list, list]:
    bal         = float(inputs.get("outstanding_balance", 25_000))
    monthly_r   = _resolve(rates, "monthly_interest_rate_pct",   cfg["fallback_monthly_rate"])
    annual_fee  = _resolve(rates, "annual_fee",                  cfg["fallback_annual_fee"])
    late_fee    = _resolve(rates, "late_payment_fee",             cfg["fallback_late_payment_fee"])
    min_pay_pct = _resolve(rates, "minimum_payment_pct",         5.0)

    assumptions = []
    if rates.get("monthly_interest_rate_pct") is None:
        assumptions.append(f"Monthly interest rate not found in policy — used indicative {monthly_r}% p.m.")
    if rates.get("annual_fee") is None:
        assumptions.append(f"Annual fee not found in policy — used indicative ₹{annual_fee}.")

    monthly_charge = bal * monthly_r / 100.0
    annual_apr     = (1 + monthly_r / 100.0) ** 12 - 1
    annual_charge  = bal * annual_apr
    min_payment    = bal * min_pay_pct / 100.0
    daily_charge   = bal * (monthly_r / 100.0) / 30.0

    results = [
        {"label": "Monthly Finance Charge",  "value": _fmt(monthly_charge),      "is_key": True,  "note": f"{monthly_r:.2f}% per month"},
        {"label": "Effective Annual Rate",   "value": f"{annual_apr*100:.2f}% p.a.", "is_key": True, "note": "APR (compounded monthly)"},
        {"label": "Annual Charge on Balance","value": _fmt(annual_charge),        "is_key": False, "note": "if balance unchanged"},
        {"label": "Daily Interest (approx.)","value": _fmt(daily_charge),         "is_key": False, "note": ""},
        {"label": "Minimum Payment Due",     "value": _fmt(min_payment),          "is_key": False, "note": f"{min_pay_pct}% of outstanding"},
        {"label": "Annual Membership Fee",   "value": _fmt(annual_fee),           "is_key": False, "note": ""},
        {"label": "Late Payment Fee",        "value": _fmt(late_fee),             "is_key": False, "note": "if minimum payment missed"},
        {"label": "Outstanding Balance",     "value": _fmt(bal),                  "is_key": False, "note": ""},
    ]
    return results, assumptions


_COMPUTE_FNS = {
    "loan_emi":    _compute_loan_emi,
    "fd_maturity": _compute_fd_maturity,
    "rd_maturity": _compute_rd_maturity,
    "savings_int": _compute_savings_int,
    "card_int":    _compute_card_int,
}


def compute_node(state: CalculatorState) -> dict:
    if state.get("error"):
        return {}

    product   = state["product"]
    cfg       = CALC_CONFIG[product]
    calc_type = cfg["calc_type"]
    rates     = state.get("rates", {})
    inputs    = state["inputs"]

    # Merge LLM-extracted notes into assumptions
    policy_notes = rates.get("notes") or []

    print(f"[Calculator] Computing {calc_type}...")
    try:
        fn = _COMPUTE_FNS[calc_type]
        results, assumptions = fn(inputs, rates, cfg)
        assumptions = list(policy_notes) + assumptions
        return {"results": results, "assumptions": assumptions}
    except Exception as exc:
        return {"error": f"Calculation error: {exc}"}


# ── Node 4: format_output ──────────────────────────────────────────────────────

def format_output_node(state: CalculatorState) -> dict:
    if state.get("error"):
        return {"summary": f"Error: {state['error']}"}

    results = state.get("results", [])
    product = state["product"]
    label   = CALC_CONFIG[product]["label"]
    inputs  = state["inputs"]

    key_items = [r for r in results if r.get("is_key")]
    if key_items:
        highlights = "  |  ".join(f"{r['label']}: {r['value']}" for r in key_items)
        summary = f"{label} — {highlights}"
    else:
        summary = f"{label}: calculation complete."

    return {"summary": summary}


# ── Graph builder ─────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(CalculatorState)
    g.add_node("retrieve_rates",  retrieve_rates_node)
    g.add_node("extract_rates",   extract_rates_node)
    g.add_node("compute",         compute_node)
    g.add_node("format_output",   format_output_node)
    g.add_edge(START,             "retrieve_rates")
    g.add_edge("retrieve_rates",  "extract_rates")
    g.add_edge("extract_rates",   "compute")
    g.add_edge("compute",         "format_output")
    g.add_edge("format_output",   END)
    return g.compile()


_graph = None


def run_calculation(product: str, inputs: dict) -> dict:
    """
    Public API: run a fee/interest calculation.

    Returns:
        product, product_label, calc_type, inputs, results, assumptions, summary, error
    """
    global _graph
    if _graph is None:
        _graph = _build_graph()

    cfg = CALC_CONFIG.get(product)
    if not cfg:
        return {
            "product": product, "product_label": product,
            "calc_type": "unknown", "inputs": inputs,
            "results": [], "assumptions": [], "summary": "",
            "error": f"Unknown product '{product}'.",
        }

    initial: CalculatorState = {
        "product":        product,
        "inputs":         inputs,
        "policy_context": "",
        "rates":          {},
        "results":        [],
        "assumptions":    [],
        "summary":        "",
        "error":          None,
    }
    result = _graph.invoke(initial)
    return {
        "product":       product,
        "product_label": cfg["label"],
        "calc_type":     cfg["calc_type"],
        "inputs":        inputs,
        "results":       result.get("results", []),
        "assumptions":   result.get("assumptions", []),
        "summary":       result.get("summary", ""),
        "error":         result.get("error"),
    }
