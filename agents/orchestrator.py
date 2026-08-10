"""
agents/orchestrator.py

LangGraph orchestrator for FinTrust Compass.

Graph topology
--------------

    [START]
       │
       ▼
  [classify]          ← Gemini LLM classifies query domain
       │
       ▼
  [route_to_agent]    ← Conditional edge → specialist node
       │
   ┌───┴────────────────────────────────────────────────┐
   ▼           ▼          ▼         ▼          ▼        ▼
[loans]  [deposits] [accounts] [cards] [compliance] [digital_banking]
   └───────────────────────────────────────────────────┘
                         │
                         ▼
                    [END]  (answer + sources in state)
"""

import json
import os
import warnings
from typing import Literal

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from agents.specialist_agents import (
    AGENT_REGISTRY,
    AccountsAgent,
    CardsAgent,
    ComplianceAgent,
    DepositsAgent,
    DigitalBankingAgent,
    LoansAgent,
)
from agents.state import AgentState
from ingestion.vector_store import get_retriever, load_vector_store

load_dotenv()

# ---------------------------------------------------------------------------
# Domain labels for the LLM classifier prompt
# ---------------------------------------------------------------------------
DOMAIN_DESCRIPTIONS = {
    "loans": "home loan, personal loan, vehicle loan, EMI, interest rate, collateral, mortgage",
    "deposits": "fixed deposit, FD, recurring deposit, RD, maturity, premature withdrawal",
    "accounts": "savings account, current account, salary account, minimum balance, KYC, account opening",
    "cards": "credit card, debit card, card limit, reward points, billing cycle, card fee, CVV",
    "compliance": "AML, KYC, regulatory, compliance, money laundering, suspicious transaction, RBI guidelines",
    "digital_banking": "internet banking, mobile banking, UPI, NEFT, RTGS, IMPS, net banking, app, OTP",
}

DOMAIN_LIST = ", ".join(DOMAIN_DESCRIPTIONS.keys())

CLASSIFIER_SYSTEM = f"""You are a query classifier for a bank employee assistant system.
Classify the user query into exactly ONE of these domains: {DOMAIN_LIST}

Domain descriptions:
{chr(10).join(f'- {k}: {v}' for k, v in DOMAIN_DESCRIPTIONS.items())}

Respond with a JSON object in this exact format (no markdown, no explanation):
{{"domain": "<domain_name>", "confidence": "high|medium|low"}}

If the query spans multiple domains, pick the most relevant one."""


# ---------------------------------------------------------------------------
# Node: classify
# ---------------------------------------------------------------------------

def classify_node(state: AgentState) -> AgentState:
    """Use Gemini to classify the query domain."""
    query = state["query"]
    print(f"\n[Orchestrator] Classifying query: \"{query[:80]}...\"" if len(query) > 80 else f"\n[Orchestrator] Classifying: \"{query}\"")

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
        request_timeout=60,
    )

    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM},
        {"role": "user", "content": f"Query: {query}"},
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        domain = parsed.get("domain", "compliance")
        confidence = parsed.get("confidence", "medium")

        if domain not in AGENT_REGISTRY:
            print(f"[Orchestrator] Unknown domain '{domain}', defaulting to 'compliance'")
            domain = "compliance"

        print(f"[Orchestrator] Domain: {domain} (confidence: {confidence})")

    except Exception as e:
        print(f"[Orchestrator] Classification error: {e}. Defaulting to 'compliance'.")
        domain = "compliance"
        confidence = "low"

    return {
        **state,
        "domain": domain,
        "confidence": confidence,
        "messages": state["messages"] + [HumanMessage(content=query)],
    }


# ---------------------------------------------------------------------------
# Node factory: specialist agents
# ---------------------------------------------------------------------------

def make_specialist_node(domain: str, retriever_k: int = 6):
    """
    Factory that returns a LangGraph node function for a given domain.
    The retriever is created fresh per invocation to avoid stale state.
    """
    agent_class = AGENT_REGISTRY[domain]

    def specialist_node(state: AgentState) -> AgentState:
        query = state["query"]
        print(f"[{domain.upper()} Agent] Processing: \"{query[:60]}...\"" if len(query) > 60 else f"[{domain.upper()} Agent] Processing: \"{query}\"")

        vector_store = load_vector_store()
        retriever = get_retriever(vector_store, domain=domain, k=retriever_k)
        agent = agent_class(retriever=retriever)

        result = agent.run(query)
        answer = result["answer"]
        sources = result["sources"]

        print(f"[{domain.upper()} Agent] Done. Retrieved {len(sources)} source chunks.")

        return {
            **state,
            "answer": answer,
            "sources": sources,
            "messages": state["messages"] + [AIMessage(content=answer)],
        }

    specialist_node.__name__ = f"{domain}_node"
    return specialist_node


# ---------------------------------------------------------------------------
# Conditional edge: route to the correct specialist node
# ---------------------------------------------------------------------------

def route_to_agent(state: AgentState) -> Literal[
    "loans", "deposits", "accounts", "cards", "compliance", "digital_banking"
]:
    domain = state.get("domain", "compliance")
    return domain  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the LangGraph multi-agent orchestration graph."""

    graph = StateGraph(AgentState)

    # Add classify node
    graph.add_node("classify", classify_node)

    # Add one specialist node per domain
    for domain in AGENT_REGISTRY:
        graph.add_node(domain, make_specialist_node(domain))

    # Edges
    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_to_agent,
        {d: d for d in AGENT_REGISTRY},
    )
    for domain in AGENT_REGISTRY:
        graph.add_edge(domain, END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Singleton compiled graph (lazy-initialised on first call)
_compiled_graph = None


def get_graph():
    """Return the compiled LangGraph (builds once, reused thereafter)."""
    global _compiled_graph
    if _compiled_graph is None:
        print("[Orchestrator] Building LangGraph...")
        _compiled_graph = build_graph()
        print("[Orchestrator] Graph ready.")
    return _compiled_graph


def run_query(query: str) -> dict:
    """
    Main entry point: run a user query through the full multi-agent pipeline.

    Args:
        query: The employee's question.

    Returns:
        dict with keys: answer (str), domain (str), sources (list), confidence (str)
    """
    graph = get_graph()

    initial_state: AgentState = {
        "messages": [],
        "query": query,
        "domain": None,
        "answer": None,
        "sources": [],
        "confidence": None,
        "error": None,
    }

    final_state = graph.invoke(initial_state)

    return {
        "answer": final_state.get("answer", ""),
        "domain": final_state.get("domain", ""),
        "sources": final_state.get("sources", []),
        "confidence": final_state.get("confidence", ""),
    }
