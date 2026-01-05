"""
LangGraph-based F150 Expert Agent with Conversational Memory.

This module implements a custom agent using LangGraph primitives (StateGraph, nodes, edges).
Unlike the high-level `create_agent` abstraction, this gives us full control over the agent loop
and makes the orchestration explicit and traceable.

The agent follows a ReAct (Reasoning + Acting) pattern:
1. Call model → LLM decides whether to use tools or respond
2. If tools called → Execute tools and loop back to model
3. If no tools → Return response to user
"""

from enum import Enum
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.config import Config
from src.tools import search_f150_manual, search_web
from src.utils.conversational_filter import create_conversational_filter_node
from src.utils.approval_node import create_approval_node
from src.graph.state import F150StateWithTokens
from src.graph.token_tracking_node import create_token_tracking_node
from src.prompts.system_prompt import F150_SYSTEM_PROMPT


class NodeName(str, Enum):
    """Enum for graph node names to prevent typos and improve maintainability."""
    PRE_FILTER = "pre_filter"
    AGENT = "agent"
    TOOLS = "tools"
    TOKEN_TRACKER = "token_tracker"
    APPROVAL_GATE = "approval_gate"


def create_f150_graph(vector_store=None):
    """
    Create a LangGraph-based F150 expert agent with conversational memory.

    Architecture:
    - State: MessagesState (manages list of conversation messages)
    - Nodes: call_model (LLM reasoning), call_tools (execute tool calls)
    - Edges: Conditional routing between model and tools
    - Checkpointer: Persists conversation history across invocations

    Args:
        vector_store: FAISS vector store for RAG (optional)
        checkpointer: LangGraph checkpointer for persistence (optional, defaults to SqliteSaver)

    Returns:
        Compiled StateGraph that can be invoked with messages
    """

    # Initialize the LLM
    llm = ChatOllama(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
        base_url=Config.get_ollama_base_url()
    )

    checkpointer = InMemorySaver()

    # Configure tools
    tools = []
    if vector_store is not None:
        from src.tools import set_vector_store
        set_vector_store(vector_store)
        tools = [search_f150_manual, search_web]

    # Bind tools to the LLM so it knows what tools are available
    llm_with_tools = llm.bind_tools(tools)

    # System message defining the agent's behavior
    system_message = SystemMessage(content=F150_SYSTEM_PROMPT)

    # Define the agent node (calls the model)
    def call_model(state: F150StateWithTokens):
        """
        Node that calls the LLM with conversation history.

        The LLM will either:
        1. Make tool calls (if it needs more information)
        2. Return a final response (if it has enough information)
        """
        messages = state["messages"]

        # Inject system message at the beginning if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [system_message] + messages

        # Call the LLM
        response = llm_with_tools.invoke(messages)

        # Return the response to be added to state
        return {"messages": [response]}

    # Define conditional edge logic for pre-filter
    def route_prefilter_output(state: F150StateWithTokens) -> Literal[NodeName.AGENT, END]:  # type: ignore[valid-type]
        """
        Route prefilter decision based on message type.

        Routing logic:
        - If bypass_agent=True → route to END (conversational response already generated)
        - Otherwise → route to NodeName.AGENT (proceed to LLM)
        """
        if state.get("bypass_agent", False):
            return END
        return NodeName.AGENT

    # Define conditional edge logic for agent node
    def route_agent_output(state: F150StateWithTokens) -> Literal[NodeName.APPROVAL_GATE, NodeName.TOKEN_TRACKER]:  # type: ignore[valid-type]
        """
        Route agent output based on whether tools are requested.

        Routing logic:
        - If last message has tool_calls → route to NodeName.APPROVAL_GATE
        - Otherwise → route to NodeName.TOKEN_TRACKER (track usage before ending)
        """
        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, route to approval gate
        if last_message.tool_calls:
            return NodeName.APPROVAL_GATE
        # Otherwise, track tokens before ending
        return NodeName.TOKEN_TRACKER

    # Define conditional edge logic for approval gate node
    def route_approval_result(state: F150StateWithTokens) -> Literal[NodeName.TOOLS, NodeName.AGENT]:  # type: ignore[valid-type]
        """
        Route approval gate result based on approval decision.

        Routing logic after approval gate:
        - If last message has tool_calls → route to NodeName.TOOLS (approved)
        - Otherwise → route to NodeName.AGENT (rejected, agent should respond)
        """
        messages = state["messages"]
        last_message = messages[-1]

        if last_message.tool_calls:
            return NodeName.TOOLS
        return NodeName.AGENT

    # Build the graph
    workflow = StateGraph(F150StateWithTokens)

    # Add nodes
    # 0. pre_filter: Intercepts conversational-only messages (no tool calls needed)
    workflow.add_node(NodeName.PRE_FILTER, create_conversational_filter_node("2018 F-150"))

    # 1. agent: Invokes LLM to decide next action
    workflow.add_node(NodeName.AGENT, call_model)

    # 2. tools: Executes any tool calls (prebuilt ToolNode handles this)
    workflow.add_node(NodeName.TOOLS, ToolNode(tools))

    # 3. token_tracker: Tracks token usage and updates state
    workflow.add_node(NodeName.TOKEN_TRACKER, create_token_tracking_node(
        context_limit=Config.CONTEXT_LIMIT,
        warning_threshold=80.0
    ))

    # 4. approval_gate: Requests human approval for tool execution (if enabled)
    workflow.add_node(NodeName.APPROVAL_GATE, create_approval_node(
        enabled=Config.TOOL_APPROVAL_ENABLED
    ))

    # Add edges
    # Entry point: always start with the pre-filter
    workflow.add_edge(START, NodeName.PRE_FILTER)

    # Conditional edge from pre_filter:
    # - If conversational-only (bypass_agent=True) → go to END (skip agent and tracking)
    # - Otherwise → go to agent
    workflow.add_conditional_edges(NodeName.PRE_FILTER, route_prefilter_output)

    # Conditional edge from agent:
    # - If tool calls → go to approval_gate
    # - If no tool calls → go to token_tracker (track usage before ending)
    workflow.add_conditional_edges(NodeName.AGENT, route_agent_output)

    # Conditional edge from approval_gate:
    # - If tool calls still present → go to tools (approved)
    # - If tool calls cleared → go back to agent (rejected)
    workflow.add_conditional_edges(NodeName.APPROVAL_GATE, route_approval_result)

    # After tools execute, always go back to agent
    workflow.add_edge(NodeName.TOOLS, NodeName.AGENT)

    # After tracking tokens, end the conversation
    workflow.add_edge(NodeName.TOKEN_TRACKER, END)

    # Compile the graph with checkpointer for conversation memory
    graph = workflow.compile(checkpointer=checkpointer)

    return graph
