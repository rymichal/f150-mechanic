"""
Extended state definitions for LangGraph agents.

This module defines custom state classes that extend MessagesState
to include additional tracking and metadata fields.
"""

from typing import TypedDict, List
from langgraph.graph import MessagesState
from langchain_core.documents import Document


class F150StateWithTokens(MessagesState):
    """
    Extended state for F150 agent that includes token tracking.

    This state class adds token tracking fields to the standard MessagesState,
    allowing the graph to track context usage across the conversation and
    persist it in the checkpointer.

    Fields:
        messages: List of conversation messages (from MessagesState)
        total_tokens: Cumulative total tokens used in the conversation
        total_prompt_tokens: Cumulative prompt tokens used
        total_completion_tokens: Cumulative completion tokens used
        context_limit: Maximum context window size in tokens
        bypass_agent: Flag used by pre-filter to skip agent (from conversational_filter)
    """
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    context_limit: int = 128000
    bypass_agent: bool = False


class F150StateWithDualContext(MessagesState):
    """
    Extended state for F150 agent with separated RAG and chat contexts.

    This state separates RAG retrieval context from conversational messages,
    preventing retrieved documents from polluting the conversation history
    while maintaining full token tracking.

    Fields:
        messages: List of conversation messages (user + assistant only, NO RAG context)

        # RAG-specific fields
        rag_context: Formatted string of retrieved documents (transient, per-query)
        retrieved_documents: Raw Document objects from vector store (for debugging)

        # Token tracking fields
        total_tokens: Cumulative total tokens used in the conversation
        total_prompt_tokens: Cumulative prompt tokens used
        total_completion_tokens: Cumulative completion tokens used
        context_limit: Maximum context window size in tokens

        # Control flow fields
        bypass_agent: Flag used by pre-filter to skip agent processing
    """
    # RAG context fields
    rag_context: str = ""
    retrieved_documents: List[Document] = []

    # Token tracking fields (maintaining backward compatibility)
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    context_limit: int = 128000

    # Control flow
    bypass_agent: bool = False
