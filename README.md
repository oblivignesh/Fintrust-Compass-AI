# 🧭 FinTrust Compass

**AI-powered internal assistant for bank employees** — built on Retrieval-Augmented Generation (RAG), a multi-agent LangGraph orchestration layer, and Gemini 2.5 Flash.

---

## What Is This?

FinTrust Compass is a knowledge and decision-support tool for bank relationship managers and officers. Instead of manually searching through dozens of policy PDFs, employees can:

- **Ask natural-language questions** about any banking product or regulation
- **Check customer eligibility** for loans, accounts, deposits, and cards — with AI reasoning and officer override
- **Generate document checklists** for any product × applicant-type combination
- **Calculate EMI, interest, and fees** using exact financial formulas backed by policy data
- **Compare two products side-by-side** to advise customers

All answers are grounded in FinTrust's official policy documents — no hallucination, no guesswork.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Frontend (port 8501)          │
│  💬 Chat │ ✅ Eligibility │ 📋 Checklist │ 🧮 Calc │ ⚖️ Compare │
└──────────────────────────┬──────────────────────────────┘
                           │  HTTP / SSE
┌──────────────────────────▼──────────────────────────────┐
│                  FastAPI Backend (port 8000)              │
│  /query/stream  /eligibility  /checklist  /calculator    │
│  /compare       /health       /domains                   │
└────────┬───────────────────────────────┬────────────────┘
         │                               │
┌────────▼────────┐            ┌─────────▼────────────────┐
│  LangGraph      │            │  ChromaDB Vector Store    │
│  Multi-Agent    │◄──RAG──────│  1,245 chunks · 12 PDFs  │
│  Orchestrator   │            │  Gemini Embedding (3072d) │
└────────┬────────┘            └──────────────────────────┘
         │
┌────────▼──────────────────────────────────────────────┐
│  Specialist Agents (one per domain)                    │
│  Loans · Deposits · Accounts · Cards · Compliance     │
│  Digital Banking · Eligibility · Checklist · Calc     │
│  Comparison                                           │
└───────────────────────────────────────────────────────┘
         │
┌────────▼──────────────────┐
│  Gemini 2.5 Flash (LLM)   │
│  + Arize Phoenix Tracing  │
└───────────────────────────┘
```

---

## Features

### 💬 Chat Assistant
- Streams answers token-by-token (Server-Sent Events)
- Automatically routes each question to the right specialist agent
- Displays domain badge, confidence level, and source policy chunks
- Sidebar with categorised sample questions

### ✅ Product Eligibility Checker
- Supports 10 products: Home Loan, Personal Loan, Vehicle Loan, FD, RD, Savings/Current/Salary Account, Credit Card, Debit Card
- Dynamic profile form adapts fields to the selected product
- AI returns **PASS / CONDITIONAL / FAIL** with per-criterion breakdown
- **Human-in-the-Loop (HITL) Officer Review** — officer can accept, override, or escalate the AI decision before saving

### 📋 Document Checklist Generator
- Select product + applicant type (12 categories: Salaried, Self-Employed, NRI, HUF, Company, etc.)
- Returns a categorised checklist (Identity, Address, Income, Property, Business docs, etc.)
- Each item shows mandatory/optional flag, acceptable alternatives, and policy notes
- Download checklist as `.txt`

### 🧮 Fee & Interest Calculator
- Supports loans (EMI, amortisation schedule, processing fee), FD (maturity with compounding), RD, and Credit Card interest
- Pure arithmetic — no LLM for numbers, only for policy note extraction
- Period-wise breakdown table + policy fee notes from RAG
- Download results as `.txt`

### ⚖️ Policy Comparison Tool
- Select any two products for side-by-side comparison
- 7 pre-defined quick-pick pairs (FD vs RD, Home Loan vs Personal Loan, etc.)
- AI extracts 10–16 consistent parameters from policy docs for both products
- Colour-coded winner per parameter, score tally, key highlights, and recommendation
- Download comparison as `.txt`

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 2.5 Flash (`gemini-2.5-flash`) |
| Embeddings | Gemini Embedding (`models/gemini-embedding-001`, 3072-dim) |
| Vector Store | ChromaDB (local persist, domain-filtered retrieval) |
| Agent Framework | LangGraph `StateGraph` |
| LLM Orchestration | LangChain (langchain-google-genai, langchain-chroma) |
| API Backend | FastAPI + Uvicorn (SSE streaming) |
| Frontend | Streamlit |
| Observability | Arize Phoenix (traces all LLM + retrieval calls) |
| Evaluation | Custom RAGAS-style evaluator using Gemini as judge |

---

## Project Structure

```
Fintrust compass/
├── Knowledge_Source/          # 12 source policy PDFs
├── ingestion/
│   ├── pdf_loader.py          # Load & split PDFs, attach domain/product metadata
│   ├── embedder.py            # Rate-limit-safe Gemini embedding adapter
│   └── vector_store.py        # ChromaDB build/load + domain-filtered retriever
├── agents/
│   ├── state.py               # Shared LangGraph TypedDict state
│   ├── base_agent.py          # Abstract base class for all specialist agents
│   ├── specialist_agents.py   # 6 domain agents (Loans, Deposits, Accounts, Cards, Compliance, Digital)
│   ├── orchestrator.py        # LangGraph router — classifies query → dispatches to agent
│   ├── eligibility_agent.py   # 3-node eligibility pipeline with HITL support
│   ├── checklist_agent.py     # 3-node document checklist generator
│   ├── fee_calculator_agent.py# 3-node fee/interest calculator
│   └── comparison_agent.py    # 3-node policy comparison tool
├── api/
│   ├── main.py                # FastAPI app — all endpoints + SSE streaming
│   └── models.py              # Pydantic request/response models
├── frontend/
│   └── app.py                 # Streamlit UI — 5 tabs
├── evaluation/
│   ├── ragas_eval.py          # Custom RAGAS evaluator (Gemini judge)
│   └── test_dataset.json      # 24 Q&A pairs with ground truth
├── observability/
│   └── tracing.py             # Arize Phoenix setup + LangChain auto-instrumentation
├── chroma_db/                 # Persisted vector store (auto-created on first run)
├── ingest.py                  # CLI entry point for ingestion
├── start.sh                   # One-command startup script
└── .env                       # API keys and configuration
```

---

## How It Works

### 1. Ingestion (run once)
```bash
python ingest.py
```
- Loads 12 policy PDFs using PyMuPDF
- Splits into ~1,245 chunks (size 1,000, overlap 150)
- Attaches `domain` and `product` metadata to each chunk
- Embeds with Gemini and persists to ChromaDB

### 2. Startup
```bash
bash start.sh
# or manually:
uvicorn api.main:app --reload &
streamlit run frontend/app.py
```

### 3. Query Flow (Chat)
```
User question
    → FastAPI /query/stream
        → LangGraph Orchestrator: Gemini classifies domain (loans/deposits/accounts/cards/compliance/digital_banking)
        → Specialist Agent: ChromaDB retrieves top-6 domain-filtered chunks
        → Gemini generates answer grounded in retrieved context
        → Tokens stream back to Streamlit via SSE
```

### 4. Eligibility Flow
```
Product + Applicant Profile
    → /eligibility
        → Node 1: RAG retrieves eligibility criteria (k=8)
        → Node 2: Gemini evaluates each criterion → PASS/CONDITIONAL/FAIL
        → Node 3: Post-processing filters out "missing data" false-positives
        → Returns structured verdict + officer override UI
```

### 5. Comparison Flow
```
Product A + Product B
    → /compare
        → Node 1: RAG retrieves policy for both products in parallel (k=8 each)
        → Node 2: Gemini extracts consistent parameters for both and picks winners
        → Returns table + highlights + recommendation
```

---

## Configuration (`.env`)

```env
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_LLM_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
CHROMA_PERSIST_DIR=./chroma_db
KNOWLEDGE_SOURCE_DIR=./Knowledge_Source
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVAL_K=6
ENABLE_PHOENIX_TRACING=true
PHOENIX_PORT=6006
```

---

## Services

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI (REST + Swagger) | http://localhost:8000/docs |
| Arize Phoenix Traces | http://localhost:6006 |

---

## Evaluation

Run the custom RAGAS-style evaluation (uses Gemini as judge):

```bash
python -W ignore evaluation/ragas_eval.py --limit 24
```

Metrics: **Faithfulness**, **Answer Relevancy**, **Context Precision**, **Context Recall** — averaged per domain across 24 hand-crafted Q&A pairs.
