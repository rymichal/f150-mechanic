"""
Comparison of Tool Calling: Ollama vs Anthropic

This demonstrates the key differences in how different LLM providers
handle tool/function calling in LangChain.
"""

from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import json


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"Weather in {location}: Sunny, 72°F"


def demonstrate_ollama_tool_calling():
    """
    Ollama Tool Calling:
    - Uses OpenAI function calling format under the hood
    - Not all Ollama models support function calling!
    - Models that support it: llama3.1, llama3.2, mistral, qwen2.5, etc.
    - Models convert tools to JSON schema
    - May hallucinate tool calls if model isn't well-trained
    """
    print("=" * 60)
    print("OLLAMA TOOL CALLING")
    print("=" * 60)

    # Create LLM with tool binding
    llm = ChatOllama(
        model="llama3.2",  # Must be a tool-capable model
        temperature=0,
    )

    # Bind the tool to the LLM
    llm_with_tools = llm.bind_tools([get_weather])

    # Invoke with a message
    message = HumanMessage(content="What's the weather in Tokyo?")
    response = llm_with_tools.invoke([message])

    print(f"Response type: {type(response)}")
    print(f"Has tool calls: {hasattr(response, 'tool_calls')}")

    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"\nTool calls requested: {len(response.tool_calls)}")
        for tool_call in response.tool_calls:
            print(f"  - Tool: {tool_call['name']}")
            print(f"    Args: {tool_call['args']}")
    else:
        print(f"\nDirect response (no tool call):")
        print(f"  {response.content}")

    print("\n📚 Key Points:")
    print("  • Ollama uses OpenAI's function calling format")
    print("  • Tools are converted to JSON schema in system prompt")
    print("  • Model must explicitly support tool calling")
    print("  • Smaller models may struggle with complex tools")


def demonstrate_anthropic_tool_calling():
    """
    Anthropic Tool Calling:
    - Native tool use support built into Claude
    - More reliable and consistent than open models
    - Uses Anthropic's proprietary tool use format
    - Better at deciding when NOT to use tools
    - Supports parallel tool calling
    """
    print("\n" + "=" * 60)
    print("ANTHROPIC TOOL CALLING")
    print("=" * 60)

    # Create LLM with tool binding
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0,
    )

    # Bind the tool to the LLM
    llm_with_tools = llm.bind_tools([get_weather])

    # Invoke with a message
    message = HumanMessage(content="What's the weather in Tokyo?")
    response = llm_with_tools.invoke([message])

    print(f"Response type: {type(response)}")
    print(f"Has tool calls: {hasattr(response, 'tool_calls')}")

    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"\nTool calls requested: {len(response.tool_calls)}")
        for tool_call in response.tool_calls:
            print(f"  - Tool: {tool_call['name']}")
            print(f"    Args: {tool_call['args']}")
    else:
        print(f"\nDirect response (no tool call):")
        print(f"  {response.content}")

    print("\n📚 Key Points:")
    print("  • Claude has native tool use in its API")
    print("  • More accurate at parameter extraction")
    print("  • Better at multi-step reasoning about tools")
    print("  • Supports parallel tool calling out of the box")


def show_raw_api_differences():
    """
    Show the actual API format differences
    """
    print("\n" + "=" * 60)
    print("RAW API FORMAT COMPARISON")
    print("=" * 60)

    # Ollama format (OpenAI-style)
    ollama_format = {
        "model": "llama3.2",
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
    }

    # Anthropic format
    anthropic_format = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        ]
    }

    print("\n🔵 Ollama (OpenAI format):")
    print(json.dumps(ollama_format, indent=2))

    print("\n🟣 Anthropic format:")
    print(json.dumps(anthropic_format, indent=2))

    print("\n📚 Key Differences:")
    print("  • Ollama wraps function in 'function' object")
    print("  • Anthropic uses 'input_schema' instead of 'parameters'")
    print("  • Anthropic format is flatter and more direct")


def show_langchain_abstraction():
    """
    Show how LangChain abstracts these differences
    """
    print("\n" + "=" * 60)
    print("HOW LANGCHAIN ABSTRACTS THE DIFFERENCES")
    print("=" * 60)

    print("""
LangChain's @tool decorator creates a universal tool definition:

    @tool
    def get_weather(location: str) -> str:
        '''Get weather for a location.'''
        return f"Weather in {location}"

This gets converted to:
    1. OpenAI format for Ollama
    2. Anthropic format for Claude
    3. Other formats for other providers

The `.bind_tools()` method handles the conversion!

Benefits:
    ✓ Write tools once, use with any provider
    ✓ Switch providers without changing tool code
    ✓ Provider-specific optimizations handled automatically
    ✓ Consistent interface regardless of backend

Challenges:
    ✗ Not all models support tools equally well
    ✗ Some advanced features may not work across all providers
    ✗ Performance varies significantly by model
    """)


if __name__ == "__main__":
    print("\n🎓 TOOL CALLING: OLLAMA vs ANTHROPIC DEEP DIVE\n")

    # Note: These would require API keys to actually run
    print("⚠️  This is a demonstration script showing the concepts.")
    print("⚠️  Uncomment sections to test with your actual setup.\n")

    # Uncomment to test with your Ollama instance:
    # demonstrate_ollama_tool_calling()

    # Uncomment to test with Anthropic (requires API key):
    # demonstrate_anthropic_tool_calling()

    # Show the raw formats
    show_raw_api_differences()

    # Show LangChain abstraction
    show_langchain_abstraction()
