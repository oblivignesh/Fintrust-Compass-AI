"""
agents/state.py

Shared LangGraph state definition used by every node in the graph.
All agents read from and write to this TypedDict.
"""

from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Shared state passed between LangGraph nodes.

    Fields
    ------
    messages     : Full conversation history (human + AI turns).
                   Uses add_messages reducer — appends, never overwrites.
    query        : The raw user query string (extracted from the last human msg).
    domain       : Domain detected by the orchestrator (e.g. 'loans', 'cards').
    answer       : Final answer produced by the specialist agent.
    sources      : List of source metadata dicts from retrieved chunks.
    confidence   : Optional confidence label set by the orchestrator.
    error        : Optional error message if a node fails.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    domain: Optional[str]
    answer: Optional[str]
    sources: Optional[List[dict]]
    confidence: Optional[str]
    error: Optional[str]
