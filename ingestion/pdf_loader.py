"""
ingestion/pdf_loader.py

Loads all PDFs from the Knowledge_Source directory, splits them into chunks,
and attaches domain metadata so agents can filter by topic.
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Domain mapping — maps a filename keyword to a semantic domain tag
# ---------------------------------------------------------------------------
DOMAIN_MAP = {
    "Home_Loan": "loans",
    "Personal_Loan": "loans",
    "Vehicle_Loan": "loans",
    "Fixed_Deposit": "deposits",
    "Recurring_Deposit": "deposits",
    "Savings_Account": "accounts",
    "Current_Account": "accounts",
    "Salary_Account": "accounts",
    "Credit_Card": "cards",
    "Debit_Card": "cards",
    "Compliance": "compliance",
    "Digital_Banking": "digital_banking",
}


def _infer_domain(filename: str) -> str:
    """Return the domain tag for a PDF filename."""
    for keyword, domain in DOMAIN_MAP.items():
        if keyword.lower() in filename.lower():
            return domain
    return "general"


def _infer_product(filename: str) -> str:
    """Return a short product label derived from the filename stem."""
    stem = Path(filename).stem  # e.g. "FinTrust_Home_Loan_Policy"
    # Strip the leading "FinTrust_" prefix and trailing "_Policy" / "_Terms..."
    product = stem.replace("FinTrust_", "").replace("_Policy", "").replace("_Terms_and_Conditions", "")
    return product.replace("_", " ")


def load_documents(pdf_dir: str) -> List[Document]:
    """
    Load every PDF in *pdf_dir* using PyMuPDF and return raw LangChain Documents.
    Each document page carries metadata: source, domain, product, page.
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Knowledge source directory not found: {pdf_dir}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")

    all_docs: List[Document] = []
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyMuPDFLoader(str(pdf_path))
        pages = loader.load()

        domain = _infer_domain(pdf_path.name)
        product = _infer_product(pdf_path.name)

        for page in pages:
            page.metadata["domain"] = domain
            page.metadata["product"] = product
            page.metadata["source_file"] = pdf_path.name

        all_docs.extend(pages)

    print(f"\n  Total pages loaded: {len(all_docs)}")
    return all_docs


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split loaded pages into smaller overlapping chunks for retrieval.
    Metadata from the original page is preserved on every chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # Add a chunk index to metadata for debugging / tracing
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    print(f"  Total chunks after splitting: {len(chunks)}")
    return chunks


def load_and_split(
    pdf_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Convenience wrapper: load PDFs and split in one call."""
    print(f"\n[PDF Loader] Loading PDFs from: {pdf_dir}")
    docs = load_documents(pdf_dir)
    print(f"[PDF Loader] Splitting into chunks (size={chunk_size}, overlap={chunk_overlap})")
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunks
