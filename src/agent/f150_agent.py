from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from ..config import Config
from ..tools import search_f150_manual, search_web
from ..prompts.system_prompt import F150_SYSTEM_PROMPT


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
        tools = [search_f150_manual, search_web]
    else:
        tools = []

    # Create the agent using LangChain's agent with centralized system prompt
    agent = create_agent(
        llm,
        tools,
        checkpointer=InMemorySaver(),
        system_prompt=F150_SYSTEM_PROMPT
    )

    return agent
