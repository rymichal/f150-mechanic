"""
Chat Agent Node for conversational response generation.

This node handles the response generation phase, using RAG context
injected via SystemMessage without polluting conversation history.
"""

from typing import Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from src.graph.state import F150StateWithDualContext
from src.config import Config


def create_chat_agent_node(llm: ChatOllama, base_system_prompt: str):
    """
    Factory function to create a Chat agent node.

    This node generates conversational responses using:
    - Base system prompt (F150 expert persona)
    - RAG context (injected transiently)
    - Conversation history (clean, no RAG pollution)
    - Web search tool (for non-manual queries)

    Args:
        llm: ChatOllama instance with tools bound
        base_system_prompt: Base system prompt for F150 expert

    Returns:
        A node function for the LangGraph workflow
    """

    def chat_agent_node(state: F150StateWithDualContext) -> Dict:
        """
        Chat Agent node that generates conversational responses.

        Context Injection Strategy:
        - Creates a TEMPORARY SystemMessage with RAG context
        - Injects it into the prompt for THIS inference only
        - Does NOT add it to state["messages"]
        - This prevents RAG context from accumulating in history

        Args:
            state: Current graph state

        Returns:
            Dict with new assistant message
        """
        if Config.TELEMETRY:
            print("\n🤖 AGENT: Processing query and deciding on actions...")

        messages = state["messages"]
        rag_context = state.get("rag_context", "")

        # Build the prompt with RAG context injection
        prompt_messages = _build_chat_prompt(
            messages=messages,
            rag_context=rag_context,
            base_system_prompt=base_system_prompt
        )

        if Config.TELEMETRY:
            has_rag = bool(rag_context)
            print(f"  RAG context available: {has_rag}")

        # Call LLM with context-injected prompt
        response = llm.invoke(prompt_messages)

        if Config.TELEMETRY:
            has_tool_calls = bool(response.tool_calls)
            if has_tool_calls:
                tool_names = [tc.get('name') for tc in response.tool_calls]
                print(f"  ✓ Agent decided to call tools: {tool_names}")
            else:
                print("  ✓ Agent generated final response (no tool calls)")

        # Return ONLY the assistant's message (no RAG context)
        return {"messages": [response]}

    return chat_agent_node


def _build_chat_prompt(
    messages: list,
    rag_context: str,
    base_system_prompt: str
) -> list:
    """
    Build the full prompt with RAG context injected via SystemMessage.

    This is the key mechanism for separating RAG context from chat history.
    The RAG context is injected as a SystemMessage ONLY for this inference,
    but is NOT added to state["messages"].

    Args:
        messages: Conversation history (user + assistant messages only)
        rag_context: Formatted RAG context from retrieval
        base_system_prompt: Base system prompt defining agent behavior

    Returns:
        List of messages for LLM invocation
    """
    # Start with base system prompt
    system_content = base_system_prompt

    # Inject RAG context if available
    if rag_context:
        system_content = f"""{base_system_prompt}

{rag_context}

IMPORTANT: Use the retrieved context above to answer the user's question accurately.
Reference page numbers when citing information from the manual."""

    # Build prompt: SystemMessage + conversation history
    prompt_messages = [SystemMessage(content=system_content)]

    # Add conversation history (this is clean - no RAG context)
    # Check if first message is already a SystemMessage
    if messages and isinstance(messages[0], SystemMessage):
        # Skip the first system message from history
        prompt_messages.extend(messages[1:])
    else:
        prompt_messages.extend(messages)

    return prompt_messages
