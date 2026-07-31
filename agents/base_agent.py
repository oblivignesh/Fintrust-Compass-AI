"""
agents/base_agent.py

Base class for all FinTrust specialist agents.

Every specialist agent:
  1. Receives a filtered retriever (only its domain's chunks)
  2. Builds a RAG chain using Gemini LLM
  3. Returns answer + source metadata
"""

import os
import warnings
from abc import ABC, abstractmethod
from typing import List

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ---------------------------------------------------------------------------
# Shared system prompt template used by every specialist agent
# ---------------------------------------------------------------------------
_BASE_SYSTEM_PROMPT = """You are a knowledgeable bank employee assistant for FinTrust Bank, \
specialising in {domain_label}.

Use ONLY the information provided in the context below to answer the employee's question. \
If the answer cannot be found in the context, clearly state that the information is not available \
in the current policy documents — do NOT make up information.

Always be precise, professional, and cite the specific policy section or document when possible.

Context:
{context}"""

_HUMAN_PROMPT = "{question}"


def _format_docs(docs) -> str:
    """Convert retrieved documents into a single context string."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source_file', 'Unknown')} | "
        f"Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def _get_llm() -> ChatGoogleGenerativeAI:
    """Return a Gemini LLM instance shared across all agents."""
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1,
        max_output_tokens=2048,
    )


class BaseSpecialistAgent(ABC):
    """
    Abstract base class for specialist agents.
    Subclasses only need to define domain, domain_label, and optionally
    override build_prompt().
    """

    domain: str = ""          # matches metadata 'domain' field in ChromaDB
    domain_label: str = ""    # human-readable name for prompts

    def __init__(self, retriever: VectorStoreRetriever):
        self.retriever = retriever
        self.llm = _get_llm()
        self._chain = self._build_chain()

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", _BASE_SYSTEM_PROMPT.format(
                domain_label=self.domain_label,
                context="{context}"
            )),
            ("human", _HUMAN_PROMPT),
        ])

    def _build_chain(self):
        """Build a RAG chain: retrieve → format → prompt → LLM → parse."""
        prompt = self._build_prompt()
        return (
            {
                "context": self.retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def run(self, query: str) -> dict:
        """
        Execute the RAG chain for the given query.

        Returns
        -------
        dict with keys:
            answer  : str — LLM response
            sources : List[dict] — source metadata for each retrieved chunk
        """
        retrieved_docs = self.retriever.invoke(query)
        answer = self._chain.invoke(query)

        sources = [
            {
                "source_file": doc.metadata.get("source_file", ""),
                "product": doc.metadata.get("product", ""),
                "domain": doc.metadata.get("domain", ""),
                "page": doc.metadata.get("page", ""),
                "snippet": doc.page_content[:200],
            }
            for doc in retrieved_docs
        ]

        return {"answer": answer, "sources": sources}
