"""
Agentic RAG Agent Node for intelligent document retrieval and context preparation.

This node implements an agentic RAG system that uses an LLM to:
- Reformulate queries for better retrieval
- Assess relevance of retrieved documents
- Iteratively search if needed
"""

from typing import Dict, List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from src.graph.state import F150StateWithDualContext
from src.config import Config


def create_agentic_rag_node(vector_store=None, llm=None):
    """
    Factory function to create an agentic RAG node.

    This node uses an LLM to intelligently retrieve documents:
    - Query reformulation for better semantic search
    - Relevance assessment of retrieved chunks
    - Iterative retrieval if initial results insufficient

    Args:
        vector_store: FAISS vector store for retrieval
        llm: ChatOllama instance for agentic reasoning (optional, creates new if None)

    Returns:
        A node function for the LangGraph workflow
    """

    # Create a small, fast LLM for RAG operations if not provided
    if llm is None:
        llm = ChatOllama(
            model="llama3.2:latest",  # Use same model for consistency
            temperature=0,  # Deterministic for RAG
            base_url=Config.get_ollama_base_url()
        )

    def agentic_rag_node(state: F150StateWithDualContext) -> Dict:
        """
        Agentic RAG node that intelligently retrieves documents.

        Strategy: LLM-powered agentic retrieval
        - Extracts query from tool call
        - Uses LLM to reformulate query for better retrieval
        - Performs vector similarity search
        - Assesses relevance and iterates if needed
        - Returns formatted context (NOT added to messages)

        Args:
            state: Current graph state

        Returns:
            Dict with rag_context, retrieved_documents, and ToolMessage
        """
        messages = state["messages"]

        # Extract the search_f150_manual tool call
        query, tool_call_id = _extract_rag_tool_call(messages)

        if not query:
            return {
                "rag_context": "",
                "retrieved_documents": [],
                "messages": [ToolMessage(
                    content="Error: No search query found",
                    tool_call_id=tool_call_id or "unknown"
                )]
            }

        if Config.TELEMETRY:
            print("\n📚 AGENTIC_RAG: Retrieving and processing documents...")

        if vector_store is None:
            return {
                "rag_context": "",
                "retrieved_documents": [],
                "messages": [ToolMessage(
                    content="Error: Manual not loaded",
                    tool_call_id=tool_call_id
                )]
            }

        # Step 1: Reformulate query using LLM for better retrieval
        reformulated_query = _reformulate_query(query, llm)
        if Config.TELEMETRY:
            print(f"  Query: {query}")
            if reformulated_query != query:
                print(f"  Reformulated: {reformulated_query}")

        # Step 2: Retrieve documents with reformulated query
        results = vector_store.similarity_search(reformulated_query, k=5)

        # Step 3: Assess relevance and iterate if needed
        relevant_docs = _assess_relevance(results, query, llm)
        if Config.TELEMETRY:
            print(f"  Found {len(relevant_docs)} relevant chunks")

        # Step 4: If insufficient relevant docs, try again with original query
        if len(relevant_docs) < 2 and reformulated_query != query:
            if Config.TELEMETRY:
                print("  Insufficient results, trying original query...")
            results = vector_store.similarity_search(query, k=8)
            relevant_docs = _assess_relevance(results, query, llm)
            if Config.TELEMETRY:
                print(f"  Found {len(relevant_docs)} relevant chunks (2nd attempt)")

        if not relevant_docs:
            tool_response = "No relevant information found in the F-150 Owner's Manual for this query."
        else:
            # Format context for Chat Agent (transient injection)
            formatted_context = _format_rag_context(relevant_docs)
            tool_response = f"Retrieved {len(relevant_docs)} relevant sections from the manual."

        if Config.TELEMETRY:
            print(f"  ✓ Agentic RAG complete - context prepared")

        # Return tool result + rag_context (separate from messages)
        return {
            "rag_context": formatted_context if relevant_docs else "",
            "retrieved_documents": relevant_docs,
            "messages": [ToolMessage(
                content=tool_response,
                tool_call_id=tool_call_id
            )]
        }

    return agentic_rag_node


def _extract_rag_tool_call(messages: list) -> tuple[str, str]:
    """
    Extract the search query from the search_f150_manual tool call.

    Args:
        messages: List of conversation messages

    Returns:
        Tuple of (query, tool_call_id)
    """
    # Find the last AI message with tool calls
    for msg in reversed(messages):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.get('name') == 'search_f150_manual':
                    query = tool_call.get('args', {}).get('question', '')
                    tool_call_id = tool_call.get('id', 'unknown')
                    return query, tool_call_id
    return None, None


def _reformulate_query(query: str, llm: ChatOllama) -> str:
    """
    Use LLM to reformulate the query for better semantic search.

    Args:
        query: Original user query
        llm: LLM for reformulation

    Returns:
        Reformulated query optimized for vector search
    """
    reformulation_prompt = f"""Reformulate this search query for a Ford F-150 Owner's Manual vector database.

CRITICAL RULES:
1. Return ONLY the reformulated query text (no explanations, no preambles)
2. DO NOT include "Ford" or "F-150" (database is already F-150-specific)
3. Extract core component/feature names and their attributes
4. Remove conversational words ("what", "how", "please")
5. Keep it under 10 words

Examples:
- "What is fuse 33 for?" → "fuse 33 purpose amperage"
- "How do I reset the oil light?" → "oil change indicator reset procedure"
- "What's the towing capacity?" → "maximum towing capacity"
- "Where is the spare tire?" → "spare tire location access"
- "How to check transmission fluid?" → "transmission fluid level check procedure"

Query: "{query}"
Reformulated:"""

    try:
        response = llm.invoke([HumanMessage(content=reformulation_prompt)])
        reformulated = response.content.strip()

        # Additional cleanup: remove common preamble phrases if LLM ignores instructions
        preambles = [
            "here's a reformulated query:",
            "reformulated query:",
            "here is the reformulated query:",
            "reformulated:",
        ]
        reformulated_lower = reformulated.lower()
        for preamble in preambles:
            if reformulated_lower.startswith(preamble):
                reformulated = reformulated[len(preamble):].strip()
                # Remove quotes if present
                reformulated = reformulated.strip('"\'')
                break

        return reformulated if reformulated else query
    except:
        return query  # Fallback to original on error


def _assess_relevance(documents: List[Document], query: str, llm: ChatOllama) -> List[Document]:
    """
    Use LLM to assess which retrieved documents are actually relevant.

    Args:
        documents: Retrieved documents
        query: Original query
        llm: LLM for relevance assessment

    Returns:
        List of relevant documents
    """
    if not documents:
        return []

    # For efficiency, use simple heuristic if many docs
    if len(documents) <= 5:
        return documents  # Trust top 5 from vector search

    # If we have more than 5, use LLM to filter
    relevant = []
    for doc in documents[:10]:  # Max 10 to assess
        assessment_prompt = f"""Is this manual excerpt relevant to the query?

Query: "{query}"

Excerpt: "{doc.page_content[:500]}"

Answer ONLY "YES" or "NO":"""

        try:
            response = llm.invoke([HumanMessage(content=assessment_prompt)])
            if 'yes' in response.content.lower():
                relevant.append(doc)
        except:
            relevant.append(doc)  # Include on error (conservative)

    return relevant[:5]  # Return top 5 relevant


def _format_rag_context(documents: list[Document]) -> str:
    """
    Format retrieved documents into a context string for the Chat Agent.

    Args:
        documents: List of retrieved Document objects

    Returns:
        Formatted string suitable for injection into Chat Agent prompt
    """
    if not documents:
        return ""

    context_parts = ["=== RETRIEVED CONTEXT FROM F-150 MANUAL ===\n"]

    for i, doc in enumerate(documents, 1):
        page = doc.metadata.get('page', 'unknown')
        content = doc.page_content.strip()
        context_parts.append(f"[Excerpt {i} - Page {page}]\n{content}\n")

    context_parts.append("=== END OF RETRIEVED CONTEXT ===\n")

    return "\n".join(context_parts)
