from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from ..config import Config
from ..tools import get_current_location, get_weather


def create_weather_agent():
    """
    Create and configure the weather agent using modern LangChain/LangGraph pattern.

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
    tools = [get_current_location, get_weather]

    # System message for the agent
    system_message = """You are a helpful weather assistant. You can help users check the weather for any location.

When a user asks about the weather:
1. If they don't specify a location or mention "my location", "here", or similar, use the get_current_location tool to get their default location
2. Then use the get_weather tool to fetch the weather for that location
3. Present the weather information in a friendly, conversational way

Always be helpful and provide clear, concise weather information."""

    # Create the agent using LangGraph's react agent
    # Note: The parameter name changed from 'state_modifier' to 'prompt' in newer versions
    agent = create_react_agent(llm, tools, prompt=system_message)

    return agent
