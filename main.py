import warnings

# Suppress Pydantic v1 compatibility warning in Python 3.14
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from langchain_core.messages import HumanMessage

from src.agent import create_f150_agent
from src.config import Config
from src.rag import create_vector_store


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

    # Create the vector store (Approach 1: on startup)
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

    # Interactive loop
    print("\n" + "=" * 70)
    print("2018 Ford F-150 Expert Assistant")
    print("=" * 70)
    print("Ask me anything about your 2018 F-150!")
    print("Examples:")
    print("  - What is fuse 33 for?")
    print("  - How do I check tire pressure?")
    print("  - What's the towing capacity?")
    print("  - How often should I change the oil?")
    print("\nType 'quit' to exit")
    print("=" * 70)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            # LangGraph agents expect messages format
            response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

            # Extract the final message from the agent
            final_message = response["messages"][-1].content
            print(f"\nF150 Expert: {final_message}")
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
