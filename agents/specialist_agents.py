"""
agents/specialist_agents.py

Concrete specialist agents — one per banking domain.
Each agent subclasses BaseSpecialistAgent and gets a filtered retriever
so it only searches its own slice of the ChromaDB collection.
"""

from agents.base_agent import BaseSpecialistAgent
from langchain_core.vectorstores import VectorStoreRetriever


class LoansAgent(BaseSpecialistAgent):
    """Handles Home Loan, Personal Loan, and Vehicle Loan queries."""
    domain = "loans"
    domain_label = "Loans (Home, Personal, Vehicle)"

    def _build_prompt(self):
        from langchain_core.prompts import ChatPromptTemplate
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are a FinTrust Bank loan specialist. "
             "Answer employee questions about loan eligibility, interest rates, "
             "EMI calculations, documentation requirements, and loan processing "
             "using ONLY the provided policy context.\n\n"
             "Context:\n{context}"),
            ("human", "{question}"),
        ])


class DepositsAgent(BaseSpecialistAgent):
    """Handles Fixed Deposit and Recurring Deposit queries."""
    domain = "deposits"
    domain_label = "Deposits (Fixed & Recurring)"

    def _build_prompt(self):
        from langchain_core.prompts import ChatPromptTemplate
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are a FinTrust Bank deposits specialist. "
             "Answer employee questions about FD and RD schemes, interest rates, "
             "tenure options, premature withdrawal penalties, and maturity rules "
             "using ONLY the provided policy context.\n\n"
             "Context:\n{context}"),
            ("human", "{question}"),
        ])


class AccountsAgent(BaseSpecialistAgent):
    """Handles Savings, Current, and Salary Account queries."""
    domain = "accounts"
    domain_label = "Accounts (Savings, Current, Salary)"

    def _build_prompt(self):
        from langchain_core.prompts import ChatPromptTemplate
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are a FinTrust Bank accounts specialist. "
             "Answer employee questions about account opening procedures, minimum "
             "balance requirements, charges, KYC documents, and account features "
             "using ONLY the provided policy context.\n\n"
             "Context:\n{context}"),
            ("human", "{question}"),
        ])


class CardsAgent(BaseSpecialistAgent):
    """Handles Credit Card and Debit Card queries."""
    domain = "cards"
    domain_label = "Cards (Credit & Debit)"

    def _build_prompt(self):
        from langchain_core.prompts import ChatPromptTemplate
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are a FinTrust Bank cards specialist. "
             "Answer employee questions about card eligibility, credit limits, "
             "annual fees, reward points, billing cycles, and card dispute procedures "
             "using ONLY the provided policy context.\n\n"
             "Context:\n{context}"),
            ("human", "{question}"),
        ])


class ComplianceAgent(BaseSpecialistAgent):
    """Handles regulatory compliance and AML/KYC queries."""
    domain = "compliance"
    domain_label = "Regulatory Compliance"

    def _build_prompt(self):
        from langchain_core.prompts import ChatPromptTemplate
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are a FinTrust Bank compliance officer assistant. "
             "Answer employee questions about AML policies, KYC requirements, "
             "regulatory guidelines, reporting obligations, and internal controls "
             "using ONLY the provided compliance manual context. "
             "Always emphasise the importance of adherence to regulations.\n\n"
             "Context:\n{context}"),
            ("human", "{question}"),
        ])


class DigitalBankingAgent(BaseSpecialistAgent):
    """Handles digital banking, internet/mobile banking, and UPI queries."""
    domain = "digital_banking"
    domain_label = "Digital Banking"

    def _build_prompt(self):
        from langchain_core.prompts import ChatPromptTemplate
        return ChatPromptTemplate.from_messages([
            ("system",
             "You are a FinTrust Bank digital banking specialist. "
             "Answer employee questions about internet banking, mobile app, UPI, "
             "NEFT/RTGS/IMPS, two-factor authentication, and digital service "
             "policies using ONLY the provided policy context.\n\n"
             "Context:\n{context}"),
            ("human", "{question}"),
        ])


# ---------------------------------------------------------------------------
# Registry — maps domain string → agent class
# ---------------------------------------------------------------------------
AGENT_REGISTRY: dict[str, type[BaseSpecialistAgent]] = {
    "loans": LoansAgent,
    "deposits": DepositsAgent,
    "accounts": AccountsAgent,
    "cards": CardsAgent,
    "compliance": ComplianceAgent,
    "digital_banking": DigitalBankingAgent,
}
