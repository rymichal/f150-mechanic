"""
RAG retrieval tool for searching the F150 owner's manual.

This tool enables the agent to search the manual using semantic similarity,
finding relevant information even when exact keywords don't match.
"""

from langchain.tools import tool
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS

# Global variable to store the vector store
# This is set when the agent is created
_vector_store = None


def set_vector_store(vector_store: "FAISS"):
    """
    Set the global vector store for the manual search tool.

    This must be called before the tool can be used.

    Args:
        vector_store: FAISS vector store with indexed F150 manual chunks
    """
    global _vector_store
    _vector_store = vector_store


@tool
def search_f150_manual(question: str) -> str:
    """
    Search the 2018 Ford F-150 Owner's Manual for information.

    Use this tool when you need specific information from the owner's manual about:
    - Fuse locations and purposes
    - Vehicle features and controls
    - Maintenance procedures
    - Specifications and capacities
    - Safety systems
    - Troubleshooting
    - Audio/climate/infotainment systems

    Args:
        question: The question or topic to search for in the manual

    Returns:
        Relevant excerpts from the owner's manual that answer the question

    Example:
        >>> search_f150_manual("What is fuse 33 for?")
        "According to the manual, fuse 33 (15A) is for..."
    """
    if _vector_store is None:
        return "Error: Manual not loaded. Please restart the application."

    # Search for relevant chunks (top 5 most similar)
    results = _vector_store.similarity_search(question, k=5)

    if not results:
        return "No relevant information found in the manual for this question."

    # Format the results for the LLM
    formatted_results = []
    formatted_results.append("Here are the relevant sections from the F-150 Owner's Manual:\n")

    for i, doc in enumerate(results, 1):
        page = doc.metadata.get('page', 'unknown')
        content = doc.page_content.strip()

        formatted_results.append(f"\n[Section {i} - Page {page}]")
        formatted_results.append(content)
        formatted_results.append("-" * 50)

    return "\n".join(formatted_results)
