import warnings
import os

# Suppress Pydantic v1 compatibility warning in Python 3.14
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from langchain_core.messages import HumanMessage

from src.agent import create_f150_agent
from src.config import Config
from src.rag import create_vector_store
from src.utils.token_counter_chain import OllamaTokenCounter, extract_and_display_token_usage


def setup_langsmith_tracing():
    """
    Configure LangSmith tracing using environment variables.

    LangGraph/LangChain automatically detect these environment variables
    and enable tracing without requiring code changes to the agent.
    """
    if Config.LANGSMITH_TRACING:
        if not Config.LANGSMITH_API_KEY:
            print("⚠️  Warning: LANGSMITH_TRACING=true but LANGSMITH_API_KEY not set")
            print("   Tracing will be disabled. Add your API key to .env file.")
            return False

        # Set environment variables that LangChain/LangGraph automatically detect
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = Config.LANGSMITH_API_KEY
        os.environ["LANGSMITH_PROJECT"] = Config.LANGSMITH_PROJECT

        print(f"✓ LangSmith tracing enabled")
        print(f"  Project: {Config.LANGSMITH_PROJECT}")
        print(f"  View traces at: https://smith.langchain.com")
        return True

    return False


def print_startup_banner():
    """Print the startup initialization banner."""
    print("=" * 70)
    print("F150 EXPERT AGENT - INITIALIZING")
    print("=" * 70)
    print()
    print("Step 1: Loading 2018 Ford F-150 Owner's Manual...")
    print("  - Reading PDF (641 pages)")
    print("  - Creating chunks (1649 chunks)")
    print("  - Generating embeddings")
    print("  ⏳ This may take 15-30 seconds...")
    print()


def print_welcome_message():
    """Print the welcome message for the interactive loop."""
    print("\n" + "=" * 70)
    print("2018 Ford F-150 Expert Assistant")
    print("=" * 70)
    print("Ask me anything about your 2018 F-150!")
    print("\nType 'quit' to exit")
    print("=" * 70)


def print_session_summary(token_counter: OllamaTokenCounter):
    """Print the session summary when exiting."""
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    summary = token_counter.get_summary()
    print(f"Total tokens used: {summary['total_tokens']:,}")
    print(f"  Prompt tokens: {summary['total_prompt_tokens']:,}")
    print(f"  Completion tokens: {summary['total_completion_tokens']:,}")
    print(f"Total interactions: {summary['total_interactions']}")
    print(f"Final context usage: {summary['usage_percentage']:.1f}%")
    print(f"Remaining tokens: {summary['remaining_tokens']:,}")
    print("=" * 70)


def read_input(token_counter):
    """
    Read user input and handle quit commands.

    Args:
        token_counter: Optional token counter for session summary

    Returns:
        User input string, or None if user wants to quit
    """
    user_input = input("\nYou: ").strip()

    if user_input.lower() in ["quit", "exit", "q"]:
        if token_counter:
            print_session_summary(token_counter)
        print("\nGoodbye!")
        return None

    return user_input


def main():
    """Run the F150 expert agent in interactive mode."""
    # Validate configuration
    if not Config.validate():
        return

    # Set up LangSmith tracing if enabled
    setup_langsmith_tracing()

    print_startup_banner()

    # Create the vector store on startup
    try:
        vector_store = create_vector_store()
        print("\n✓ Manual loaded successfully!")
    except Exception as e:
        print(f"\n❌ Error loading manual: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running on your network")
        print("  2. You have pulled the embedding model:")
        print("     ollama pull nomic-embed-text")
        return

    print("\nStep 2: Initializing AI agent...")

    # Create the agent with the vector store
    try:
        agent = create_f150_agent(vector_store)
        print("✓ Agent ready!")
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        return

    # Initialize token counter if enabled
    token_counter = None
    if Config.TOKEN_TRACKING_ENABLED:
        token_counter = OllamaTokenCounter(context_limit=Config.CONTEXT_LIMIT)
        print("\n✓ Token tracking enabled (using Ollama actual counts)")

    # Interactive loop
    print_welcome_message()

    while True:
        user_input = read_input(token_counter)

        if user_input is None:
            break

        if not user_input:
            continue

        try:
            # Invoke agent
            response = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                {"configurable": {"thread_id": "1"}}
            )

            # Extract and display the final message
            final_message_obj = response["messages"][-1]
            print(f"\nF150 Expert::::\n{final_message_obj.content}")

            # Track token usage if enabled
            if token_counter:
                extract_and_display_token_usage(token_counter, final_message_obj)

        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
