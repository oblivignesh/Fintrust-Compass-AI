"""
api/main.py

FinTrust Compass — FastAPI backend

Endpoints
---------
GET  /health          — liveness + readiness check
POST /query           — single-turn RAG query through multi-agent pipeline
POST /query/stream    — streaming version (SSE) of the same query
GET  /domains         — list available domains and their descriptions
"""

import os
import sys
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_google_genai import ChatGoogleGenerativeAI

from api.models import (
    HealthResponse, QueryRequest, QueryResponse, SourceChunk,
    EligibilityRequest, EligibilityResponse, CriterionResult,
    ChecklistRequest, ChecklistResponse, ChecklistItem,
    FeeCalcRequest, FeeCalcResponse,
    CompareRequest, CompareResponse, CompareParameter,
)
from agents.orchestrator import build_graph, run_query
from agents.eligibility_agent import check_eligibility, list_products
from agents.checklist_agent import generate_checklist, list_applicant_categories
from agents.fee_calculator_agent import calculate_fees, list_calc_products
from agents.comparison_agent import compare_products, list_comparable_products, list_comparable_groups
from ingestion.vector_store import load_vector_store
from observability.tracing import setup_tracing, get_phoenix_url

load_dotenv()

# ---------------------------------------------------------------------------
# App lifecycle: pre-load vector store and compile graph on startup
# ---------------------------------------------------------------------------

_vector_store = None
_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the vector store and LangGraph on startup."""
    global _vector_store, _graph
    print("[API] Starting Arize Phoenix tracing...")
    phoenix_url = setup_tracing(project_name="fintrust-compass")
    if phoenix_url:
        print(f"[API] Phoenix UI → {phoenix_url}")
    print("[API] Loading ChromaDB...")
    _vector_store = load_vector_store()
    print("[API] Compiling LangGraph...")
    _graph = build_graph()
    print("[API] Ready.")
    yield
    print("[API] Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FinTrust Compass API",
    description="AI-powered bank employee assistant backed by RAG + multi-agent orchestration",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DOMAIN_DESCRIPTIONS = {
    "loans": "Home Loan, Personal Loan, Vehicle Loan policies",
    "deposits": "Fixed Deposit and Recurring Deposit schemes",
    "accounts": "Savings, Current, and Salary Account policies",
    "cards": "Credit Card and Debit Card terms & conditions",
    "compliance": "AML, KYC, and regulatory compliance manual",
    "digital_banking": "Internet banking, mobile app, UPI, NEFT/RTGS policies",
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness and readiness check."""
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not ready")
    doc_count = _vector_store._collection.count()
    return HealthResponse(
        status="ok",
        vector_store_docs=doc_count,
        llm_model=os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash"),
        embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        phoenix_url=get_phoenix_url(),
    )


@app.get("/domains", tags=["System"])
async def list_domains():
    """Return all supported banking domains and their descriptions."""
    return {"domains": DOMAIN_DESCRIPTIONS}


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Run a question through the full multi-agent RAG pipeline.
    Returns the answer, domain classification, confidence, and source chunks.
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Agent graph not ready")

    try:
        result = run_query(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    sources = [SourceChunk(**s) for s in result.get("sources", [])]
    conv_id = request.conversation_id or str(uuid.uuid4())

    return QueryResponse(
        answer=result["answer"],
        domain=result["domain"],
        confidence=result["confidence"],
        sources=sources,
        conversation_id=conv_id,
    )


@app.post("/query/stream", tags=["Query"])
async def query_stream(request: QueryRequest):
    """
    Streaming version of /query using Server-Sent Events.
    Streams the LLM answer token-by-token, then sends a final JSON event
    with metadata (domain, confidence, sources).
    """
    if _graph is None:
        raise HTTPException(status_code=503, detail="Agent graph not ready")

    llm_model = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")
    api_key = os.getenv("GOOGLE_API_KEY")

    async def event_stream() -> AsyncGenerator[str, None]:
        import json
        from agents.orchestrator import DOMAIN_DESCRIPTIONS as D_DESC, CLASSIFIER_SYSTEM, route_to_agent
        from agents.specialist_agents import AGENT_REGISTRY
        from ingestion.vector_store import get_retriever
        from langchain_core.messages import HumanMessage

        # Step 1: classify domain
        llm = ChatGoogleGenerativeAI(model=llm_model, google_api_key=api_key, temperature=0)
        cls_msgs = [
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            {"role": "user", "content": f"Query: {request.question}"},
        ]
        try:
            cls_resp = llm.invoke(cls_msgs)
            raw = cls_resp.content.strip().strip("```json").strip("```").strip()
            parsed = json.loads(raw)
            domain = parsed.get("domain", "compliance")
            confidence = parsed.get("confidence", "medium")
        except Exception:
            domain, confidence = "compliance", "low"

        if domain not in AGENT_REGISTRY:
            domain = "compliance"

        yield f"data: {json.dumps({'type': 'domain', 'domain': domain, 'confidence': confidence})}\n\n"

        # Step 2: retrieve context
        retriever = get_retriever(_vector_store, domain=domain, k=int(os.getenv("RETRIEVAL_K", 6)))
        docs = retriever.invoke(request.question)
        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source_file', '')} | Page {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        )

        # Step 3: stream LLM response
        from agents.specialist_agents import AGENT_REGISTRY
        agent_class = AGENT_REGISTRY[domain]
        stream_llm = ChatGoogleGenerativeAI(
            model=llm_model, google_api_key=api_key, temperature=0.1, streaming=True
        )

        domain_labels = {
            "loans": "Loans", "deposits": "Deposits", "accounts": "Accounts",
            "cards": "Cards", "compliance": "Compliance", "digital_banking": "Digital Banking",
        }
        system_msg = (
            f"You are a FinTrust Bank employee assistant specialising in {domain_labels.get(domain, domain)}. "
            f"Answer using ONLY the context below. Be precise and professional.\n\nContext:\n{context}"
        )

        full_answer = ""
        async for chunk in stream_llm.astream([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": request.question},
        ]):
            token = chunk.content
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        # Step 4: send sources as final event
        sources = [
            {
                "source_file": d.metadata.get("source_file", ""),
                "product": d.metadata.get("product", ""),
                "domain": d.metadata.get("domain", ""),
                "page": d.metadata.get("page", ""),
                "snippet": d.page_content[:200],
            }
            for d in docs
        ]
        yield f"data: {json.dumps({'type': 'done', 'sources': sources})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Eligibility Checker endpoints
# ---------------------------------------------------------------------------

@app.get("/eligibility/products", tags=["Eligibility"])
async def eligibility_products():
    """Return the list of products supported by the eligibility checker."""
    return {"products": list_products()}


@app.post("/eligibility", response_model=EligibilityResponse, tags=["Eligibility"])
async def eligibility_check(request: EligibilityRequest):
    """
    Run a structured eligibility check for a banking product.

    The agent:
    1. Retrieves relevant eligibility criteria from policy PDFs (RAG)
    2. Evaluates each criterion against the provided applicant profile (LLM)
    3. Returns a PASS / CONDITIONAL / FAIL decision with per-criterion breakdown
    """
    try:
        result = check_eligibility(request.product, request.profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eligibility check error: {e}")

    criteria = [
        CriterionResult(
            criterion=c.get("criterion", ""),
            requirement=c.get("requirement", ""),
            applicant_value=c.get("applicant_value", "Not provided"),
            status=c.get("status", "NOT_SPECIFIED"),
            reason=c.get("reason", ""),
        )
        for c in result.get("criteria", [])
    ]

    return EligibilityResponse(
        product=result["product"],
        product_label=result["product_label"],
        decision=result["decision"],
        decision_reason=result["decision_reason"],
        conditions=result["conditions"],
        criteria=criteria,
        error=result.get("error"),
    )


# ---------------------------------------------------------------------------
# Fee and Interest Calculator endpoints
# ---------------------------------------------------------------------------

@app.get("/calculator/products", tags=["Calculator"])
async def calculator_products():
    """Return products supported by the fee/interest calculator with their input specs."""
    return {"products": list_calc_products()}


@app.post("/calculator", response_model=FeeCalcResponse, tags=["Calculator"])
async def calculator_compute(request: FeeCalcRequest):
    """
    Compute fees, EMI, interest, or maturity for a banking product.

    The agent:
    1. Retrieves current fee/rate details from policy PDFs (RAG)
    2. Performs exact arithmetic (EMI, amortisation, FD maturity, etc.)
    3. Extracts notable policy notes (charges, penalties, conditions)
    """
    try:
        result = calculate_fees(request.product, request.params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculator error: {e}")

    return FeeCalcResponse(
        product=result["product"],
        product_label=result["product_label"],
        results=result["results"],
        schedule=result["schedule"],
        policy_notes=result["policy_notes"],
        summary=result["summary"],
        error=result.get("error"),
    )


# ---------------------------------------------------------------------------
# Document Checklist Generator endpoints
# ---------------------------------------------------------------------------

@app.get("/checklist/categories", tags=["Checklist"])
async def checklist_categories():
    """Return supported applicant categories for the checklist generator."""
    return {"categories": list_applicant_categories()}


@app.post("/checklist", response_model=ChecklistResponse, tags=["Checklist"])
async def checklist_generate(request: ChecklistRequest):
    """
    Generate a structured document checklist for a banking product + applicant type.

    The agent:
    1. Retrieves document requirements from policy PDFs (RAG)
    2. Extracts a categorised, mandatory/optional checklist via LLM
    3. Returns items grouped by category with alternatives and notes
    """
    try:
        result = generate_checklist(
            request.product,
            request.applicant_category,
            request.additional_context or "",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Checklist error: {e}")

    items = [
        ChecklistItem(
            category=c.get("category", "General"),
            document=c.get("document", ""),
            mandatory=bool(c.get("mandatory", True)),
            alternatives=c.get("alternatives") or [],
            notes=c.get("notes") or "",
        )
        for c in result.get("checklist", [])
    ]

    return ChecklistResponse(
        product=result["product"],
        product_label=result["product_label"],
        applicant_category=result["applicant_category"],
        checklist=items,
        summary=result["summary"],
        error=result.get("error"),
    )


# ---------------------------------------------------------------------------
# Policy Comparison Tool endpoints
# ---------------------------------------------------------------------------

@app.get("/compare/products", tags=["Comparison"])
async def compare_products_list():
    """Return all products available for side-by-side comparison."""
    return {
        "products": list_comparable_products(),
        "quick_pairs": list_comparable_groups(),
    }


@app.post("/compare", response_model=CompareResponse, tags=["Comparison"])
async def policy_compare(request: CompareRequest):
    """
    Compare two banking products side-by-side.

    The agent:
    1. Retrieves policy chunks for both products from the vector store (RAG)
    2. Extracts a consistent set of comparable parameters via LLM
    3. Returns a structured table with per-parameter winner + highlights + recommendation
    """
    if request.product_a == request.product_b:
        raise HTTPException(status_code=400, detail="product_a and product_b must be different.")
    try:
        result = compare_products(request.product_a, request.product_b)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {e}")

    params = [
        CompareParameter(
            parameter=p.get("parameter", ""),
            value_a=p.get("value_a", "Not mentioned"),
            value_b=p.get("value_b", "Not mentioned"),
            winner=p.get("winner", "Neutral"),
            note=p.get("note", ""),
        )
        for p in result.get("parameters", [])
    ]

    return CompareResponse(
        product_a=result["product_a"],
        product_b=result["product_b"],
        label_a=result["label_a"],
        label_b=result["label_b"],
        parameters=params,
        highlights=result.get("highlights", []),
        recommendation=result.get("recommendation", ""),
        error=result.get("error"),
    )
