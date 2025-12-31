import warnings

# Suppress Pydantic v1 compatibility warning in Python 3.14
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from langchain_core.messages import HumanMessage

from src.agent import create_f150_agent
from src.config import Config
from src.rag import create_vector_store
from src.utils.token_counter import (
    OllamaTokenCounter,
    format_token_usage,
    get_progress_bar,
    get_warning_message
)


def main():
    """Run the F150 expert agent in interactive mode."""
    # Validate configuration
    if not Config.validate():
        return

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
        token_counter = OllamaTokenCounter(
            context_limit=Config.CONTEXT_LIMIT
        )
        print("\n✓ Token tracking enabled (using Ollama actual counts)")

    # Interactive loop
    print("\n" + "=" * 70)
    print("2018 Ford F-150 Expert Assistant")
    print("=" * 70)
    print("Ask me anything about your 2018 F-150!")
    print("\nType 'quit' to exit")
    print("=" * 70)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            if token_counter:
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
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        try:
            # LangGraph agents expect messages format
            response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

            # Extract the final message from the agent
            final_message_obj = response["messages"][-1]
            final_message = final_message_obj.content
            print(f"\nF150 Expert:::\n{final_message}")

            # Track token usage if enabled
            if token_counter:
                # Extract actual token counts from Ollama response
                token_counts = token_counter.extract_token_counts(final_message_obj)

                if token_counts:
                    # Use actual Ollama token counts
                    usage_stats = token_counter.track_interaction(
                        prompt_tokens=token_counts['prompt_tokens'],
                        completion_tokens=token_counts['completion_tokens']
                    )

                    print("\n" + "-" * 70)
                    print("TOKEN USAGE (Ollama actual):")
                    print(format_token_usage(usage_stats, show_details=True))

                    # Display progress bar
                    progress_bar = get_progress_bar(usage_stats['usage_percentage'])
                    print(f"\nContext: {progress_bar}")

                    # Display warning if needed
                    warning = get_warning_message(usage_stats)
                    if warning:
                        print(f"\n{warning}")

                    print("-" * 70)
                else:
                    # Ollama didn't return token counts
                    print("\n⚠️  Token tracking unavailable - Ollama did not return token counts")
                    print("    This may happen with some models or configurations.")

        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
