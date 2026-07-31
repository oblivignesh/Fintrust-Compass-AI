"""
api/models.py

Pydantic request/response models for the FinTrust Compass API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="Employee question")
    conversation_id: Optional[str] = Field(None, description="Optional session ID for multi-turn context")


class SourceChunk(BaseModel):
    source_file: str
    product: str
    domain: str
    page: str | int
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    domain: str
    confidence: str
    sources: List[SourceChunk]
    conversation_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    vector_store_docs: int
    llm_model: str
    embedding_model: str
    phoenix_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Eligibility Checker models
# ---------------------------------------------------------------------------

class EligibilityRequest(BaseModel):
    product: str = Field(
        ...,
        description=(
            "Product to check. One of: home_loan, personal_loan, vehicle_loan, "
            "fixed_deposit, recurring_deposit, savings_account, current_account, "
            "salary_account, credit_card, debit_card"
        ),
    )
    profile: dict = Field(
        ...,
        description="Applicant profile as key-value pairs (age, income, cibil_score, etc.)",
    )


class CriterionResult(BaseModel):
    criterion: str
    requirement: str
    applicant_value: str
    status: str          # PASS | CONDITIONAL | FAIL | NOT_SPECIFIED
    reason: str


class EligibilityResponse(BaseModel):
    product: str
    product_label: str
    decision: str        # PASS | CONDITIONAL | FAIL | ERROR
    decision_reason: str
    conditions: List[str]
    criteria: List[CriterionResult]
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Document Checklist Generator models
# ---------------------------------------------------------------------------

class ChecklistRequest(BaseModel):
    product: str = Field(..., description="Product ID (e.g. home_loan, credit_card)")
    applicant_category: str = Field(..., description="Applicant category (e.g. Salaried Individual)")
    additional_context: Optional[str] = Field(
        "", description="Optional extra context (e.g. 'under-construction property', 'NRI applicant')"
    )


class ChecklistItem(BaseModel):
    category: str           # e.g. "Identity Proof", "Income Documents"
    document: str           # e.g. "PAN Card"
    mandatory: bool
    alternatives: List[str] # e.g. ["Aadhaar", "Passport", "Voter ID"]
    notes: str              # e.g. "last 3 months", "original + 2 copies"


class ChecklistResponse(BaseModel):
    product: str
    product_label: str
    applicant_category: str
    checklist: List[ChecklistItem]
    summary: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Fee and Interest Calculator models
# ---------------------------------------------------------------------------

class FeeCalcRequest(BaseModel):
    product: str = Field(..., description="Product ID (e.g. home_loan, fixed_deposit, credit_card)")
    params: dict = Field(..., description="Financial parameters (principal, annual_rate, tenure_months, etc.)")


class FeeCalcResponse(BaseModel):
    product: str
    product_label: str
    results: dict                # key → formatted value string
    schedule: List[dict]         # amortisation / period-wise rows
    policy_notes: List[str]      # notable fee/charge sentences from policy docs
    summary: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Policy Comparison Tool models
# ---------------------------------------------------------------------------

class CompareRequest(BaseModel):
    product_a: str = Field(..., description="First product ID (e.g. home_loan)")
    product_b: str = Field(..., description="Second product ID (e.g. personal_loan)")


class CompareParameter(BaseModel):
    parameter: str      # e.g. "Interest Rate"
    value_a: str        # value for product A
    value_b: str        # value for product B
    winner: str         # label_a | label_b | "Neutral" | "Depends"
    note: str           # one-line explanation


class CompareResponse(BaseModel):
    product_a: str
    product_b: str
    label_a: str
    label_b: str
    parameters: List[CompareParameter]
    highlights: List[str]       # key difference bullets
    recommendation: str
    error: Optional[str] = None
