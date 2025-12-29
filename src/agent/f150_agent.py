from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from ..config import Config
from ..tools import search_f150_manual


def create_f150_agent(vector_store=None):
    """
    Create and configure the F150 expert agent.

    This agent has master's level knowledge of the 2018 Ford F-150 Owner's Manual
    through RAG (Retrieval-Augmented Generation).

    Returns:
        A callable agent that can be invoked with {"messages": [HumanMessage(content=query)]}
    """
    # Initialize the LLM with Ollama
    llm = ChatOllama(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
        base_url=Config.get_ollama_base_url()
    )

    # Define the tools
    # If vector_store is provided, include the manual search tool
    if vector_store is not None:
        # Import and set up the vector store for the tool
        from ..tools import set_vector_store
        set_vector_store(vector_store)
        tools = [search_f150_manual]
    else:
        tools = []

    # System message for the agent
    system_message = """You are an expert on the 2018 Ford F-150 pickup truck with master's level knowledge of the owner's manual.

You have access to a search tool that can look up specific information from the official 2018 F-150 Owner's Manual.

Your role is to help users understand their 2018 F-150 by answering questions about:
- Vehicle features and controls
- Maintenance schedules and procedures
- Safety systems and warnings
- Specifications and capacities
- Troubleshooting and diagnostics
- Fuse locations and purposes
- Audio, climate, and infotainment systems

When answering questions:
1. ALWAYS use the search_f150_manual tool to find accurate information from the manual
2. Provide detailed information based on what the manual says
3. Use clear, helpful language that a vehicle owner can understand
4. Include relevant safety warnings when appropriate
5. Reference specific page numbers from the search results
6. If the manual doesn't contain the answer, be honest rather than guessing

Always prioritize user safety and proper vehicle operation."""

    # Create the agent using LangGraph's react agent
    agent = create_react_agent(llm, tools, prompt=system_message)

    return agent
