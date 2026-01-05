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

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.config import Config
from src.tools import search_f150_manual, search_web
from src.utils.conversational_filter import create_conversational_filter_node
from src.graph.state import F150StateWithTokens
from src.graph.token_tracking_node import create_token_tracking_node


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
    system_message = SystemMessage(content="""You are an expert on the 2018 Ford F-150 pickup truck with master's level knowledge of the owner's manual.

You have access to TWO search tools:
1. search_f150_manual - Search the official 2018 F-150 Owner's Manual
2. search_web - Search the web for current information and real-world knowledge

Your role is to help users understand their 2018 F-150 by answering questions about:
- Vehicle features and controls
- Maintenance schedules and procedures
- Safety systems and warnings
- Specifications and capacities
- Troubleshooting and diagnostics
- Fuse locations and purposes
- Audio, climate, and infotainment systems

CRITICAL - TOOL USAGE RULES (Follow STRICTLY):
1. DO NOT call ANY tools for these messages (respond directly):
   - Greetings: "hello", "hi", "hey"
   - Thanks: "thank you", "thanks", "thx", "appreciate it"
   - Acknowledgments: "ok", "okay", "got it", "great"
   - Farewells: "bye", "goodbye", "see you"
   - Off-topic questions unrelated to the F-150

2. ONLY call tools when the user has a SPECIFIC F-150 question requiring information:
   - "What is the oil capacity?" → USE search_f150_manual
   - "How do I reset the oil light?" → USE search_f150_manual
   - "My engine is making noise" → USE BOTH tools
   - "Great thank you" → DO NOT use any tools, just respond warmly

CONVERSATIONAL HANDLING:
- For "thank you", "thanks", or similar → Respond directly: "You're welcome!" or "Happy to help!"
- For unrelated questions → Respond directly: "I'm not sure I can help with that, but I'm here to assist with any questions or problems about your 2018 F-150!"
- NO TOOLS for conversational pleasantries

TOOL SELECTION STRATEGY (Smart Routing):

Use search_f150_manual for:
- Specifications and capacities (towing, fuel, tire pressure, fluids)
- Standard operating procedures (how to use features)
- Feature explanations (what does this button do?)
- Fuse diagrams and electrical system
- Maintenance schedules from the manual
- Safety warnings and official guidance

Use search_web for:
- Known issues, recalls, and service bulletins
- Real-world troubleshooting tips
- Common problems and community solutions
- Product updates and firmware fixes
- User experiences and reviews
- Information not covered in the manual

For TROUBLESHOOTING PROBLEMS:
- Start with search_f150_manual for official guidance
- Then use search_web to find real-world fixes, known issues, and recalls
- Combine both sources for comprehensive answers

When answering questions:
1. Choose the appropriate tool(s) based on the question type
2. BEFORE using a tool, tell the user what you're doing:
   - Before search_f150_manual: "Let me check the owner's manual..."
   - Before search_web: "Let me search online for current information..."
3. For problems, use BOTH tools to provide comprehensive help
4. Provide detailed information and cite your sources
5. Use clear, helpful language that a vehicle owner can understand
6. Include relevant safety warnings when appropriate
7. Reference page numbers from manual searches
8. Distinguish between official manual guidance and web-sourced information

Always prioritize user safety and proper vehicle operation.""")

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
    def should_bypass_agent(state: F150StateWithTokens) -> Literal["agent", END]:  # type: ignore[valid-type]
        """
        Determine whether to bypass the agent (conversational-only) or proceed.

        Routing logic:
        - If bypass_agent=True → route to END (conversational response already generated)
        - Otherwise → route to "agent" (proceed to LLM)
        """
        if state.get("bypass_agent", False):
            return END
        return "agent"

    # Define conditional edge logic for agent node
    def should_continue(state: F150StateWithTokens) -> Literal["tools", "token_tracker"]:  # type: ignore[valid-type]
        """
        Determine whether to continue to tools or track tokens.

        Routing logic:
        - If last message has tool_calls → route to "tools"
        - Otherwise → route to "token_tracker" (track usage before ending)
        """
        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, route to the tools node
        if last_message.tool_calls:
            return "tools"
        # Otherwise, track tokens before ending
        return "token_tracker"

    # Build the graph
    workflow = StateGraph(F150StateWithTokens)

    # Add nodes
    # 0. pre_filter: Intercepts conversational-only messages (no tool calls needed)
    workflow.add_node("pre_filter", create_conversational_filter_node("2018 F-150"))

    # 1. agent: Invokes LLM to decide next action
    workflow.add_node("agent", call_model)

    # 2. tools: Executes any tool calls (prebuilt ToolNode handles this)
    workflow.add_node("tools", ToolNode(tools))

    # 3. token_tracker: Tracks token usage and updates state
    workflow.add_node("token_tracker", create_token_tracking_node(
        context_limit=Config.CONTEXT_LIMIT,
        warning_threshold=80.0
    ))

    # Add edges
    # Entry point: always start with the pre-filter
    workflow.add_edge(START, "pre_filter")

    # Conditional edge from pre_filter:
    # - If conversational-only (bypass_agent=True) → go to END (skip agent and tracking)
    # - Otherwise → go to agent
    workflow.add_conditional_edges("pre_filter", should_bypass_agent)

    # Conditional edge from agent:
    # - If tool calls → go to tools
    # - If no tool calls → go to token_tracker (track usage before ending)
    workflow.add_conditional_edges("agent", should_continue)

    # After tools execute, always go back to agent
    workflow.add_edge("tools", "agent")

    # After tracking tokens, end the conversation
    workflow.add_edge("token_tracker", END)

    # Compile the graph with checkpointer for conversation memory
    graph = workflow.compile(checkpointer=checkpointer)

    return graph
