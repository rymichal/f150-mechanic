"""
LangGraph-based F150 Expert Agent with Agentic RAG Architecture.

This module implements a fully agentic system where:
- The main agent decides when to use RAG (via search_f150_manual tool)
- RAG retrieval uses an agentic approach (query reformulation, relevance assessment)
- RAG context is injected transiently (doesn't pollute conversation history)

Architecture Flow:
    START → PRE_FILTER → AGENT → (decides to call search_f150_manual?) →
    APPROVAL_GATE → AGENTIC_RAG or TOOLS → AGENT (loop) → TOKEN_TRACKER → END

Key Features:
- Agent autonomously decides when RAG is needed
- Agentic RAG uses LLM for intelligent retrieval
- RAG context separate from chat history
- Sequential multi-agent pattern maintained
"""

from enum import Enum
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from src.config import Config
from src.tools import search_f150_manual, search_web
from src.utils.conversational_filter import create_conversational_filter_node
from src.utils.approval_node import create_approval_node
from src.graph.state import F150StateWithDualContext
from src.utils.token_counter_graph import create_token_tracking_node
from src.graph.rag_agent_node import create_agentic_rag_node
from src.graph.chat_agent_node import create_chat_agent_node
from src.prompts.system_prompt import F150_CHAT_AGENT_PROMPT


class NodeName(str, Enum):
    """Enum for graph node names to prevent typos and improve maintainability."""
    PRE_FILTER = "pre_filter"
    AGENT = "agent"
    AGENTIC_RAG = "agentic_rag"
    TOOLS = "tools"
    TOKEN_TRACKER = "token_tracker"
    APPROVAL_GATE = "approval_gate"


def create_f150_graph(vector_store=None):
    """
    Create a LangGraph-based F150 expert agent with agentic RAG architecture.

    Architecture:
        1. PRE_FILTER: Intercept conversational-only messages
        2. AGENT: Main agent that decides when to use RAG (via search_f150_manual tool)
        3. APPROVAL_GATE: Human approval for tool execution (if enabled)
        4. AGENTIC_RAG: Intelligent RAG retrieval (if search_f150_manual called)
        5. TOOLS: Execute other tools like search_web
        6. TOKEN_TRACKER: Track token usage

    Args:
        vector_store: FAISS vector store for RAG

    Returns:
        Compiled StateGraph
    """

    # Initialize the LLM
    llm = ChatOllama(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
        base_url=Config.get_ollama_base_url()
    )

    checkpointer = InMemorySaver()

    # Set vector store for the search_f150_manual tool
    if vector_store is not None:
        from src.tools import set_vector_store
        set_vector_store(vector_store)

    # Configure tools - BOTH manual and web search
    tools = [search_f150_manual, search_web]
    llm_with_tools = llm.bind_tools(tools)

    # System prompt for Agent
    system_prompt = F150_CHAT_AGENT_PROMPT

    # ==================== ROUTING FUNCTIONS ====================

    def route_prefilter_output(state: F150StateWithDualContext) -> Literal[NodeName.AGENT, END]:  # type: ignore[valid-type]
        """Route pre-filter decision."""
        if state.get("bypass_agent", False):
            return END
        return NodeName.AGENT

    def route_agent_output(state: F150StateWithDualContext) -> Literal[NodeName.APPROVAL_GATE, NodeName.TOKEN_TRACKER]:  # type: ignore[valid-type]
        """Route Agent output based on tool calls."""
        messages = state["messages"]
        last_message = messages[-1]

        if last_message.tool_calls:
            return NodeName.APPROVAL_GATE
        return NodeName.TOKEN_TRACKER

    def route_approval_result(state: F150StateWithDualContext) -> Literal[NodeName.AGENTIC_RAG, NodeName.TOOLS]:  # type: ignore[valid-type]
        """
        Route approval decision to appropriate tool execution node.

        - If search_f150_manual called → AGENTIC_RAG
        - Otherwise → TOOLS (for search_web, etc.)
        """
        messages = state["messages"]
        last_message = messages[-1]

        if not last_message.tool_calls:
            # No tool calls (rejected), go back to agent
            return NodeName.AGENT  # type: ignore

        # Check if any tool call is search_f150_manual
        for tool_call in last_message.tool_calls:
            if tool_call.get('name') == 'search_f150_manual':
                return NodeName.AGENTIC_RAG

        # Otherwise, use standard tools node
        return NodeName.TOOLS

    # ==================== BUILD GRAPH ====================

    workflow = StateGraph(F150StateWithDualContext)

    # Add nodes
    workflow.add_node(NodeName.PRE_FILTER, create_conversational_filter_node("2018 F-150"))
    workflow.add_node(NodeName.AGENT, create_chat_agent_node(llm_with_tools, system_prompt))
    workflow.add_node(NodeName.AGENTIC_RAG, create_agentic_rag_node(vector_store, llm))
    workflow.add_node(NodeName.TOOLS, ToolNode([search_web]))  # Only web search here
    workflow.add_node(NodeName.TOKEN_TRACKER, create_token_tracking_node(
        context_limit=Config.CONTEXT_LIMIT,
        warning_threshold=80.0
    ))
    workflow.add_node(NodeName.APPROVAL_GATE, create_approval_node(
        enabled=Config.TOOL_APPROVAL_ENABLED
    ))

    # Add edges
    workflow.add_edge(START, NodeName.PRE_FILTER)
    workflow.add_conditional_edges(NodeName.PRE_FILTER, route_prefilter_output)
    workflow.add_conditional_edges(NodeName.AGENT, route_agent_output)
    workflow.add_conditional_edges(NodeName.APPROVAL_GATE, route_approval_result)
    workflow.add_edge(NodeName.AGENTIC_RAG, NodeName.AGENT)  # Loop back to agent
    workflow.add_edge(NodeName.TOOLS, NodeName.AGENT)  # Loop back to agent
    workflow.add_edge(NodeName.TOKEN_TRACKER, END)

    # Compile
    graph = workflow.compile(checkpointer=checkpointer)

    return graph
