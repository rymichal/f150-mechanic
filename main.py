import warnings

# Suppress Pydantic v1 compatibility warning in Python 3.14
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from langchain_core.messages import HumanMessage
from langgraph.types import Command

# Use the LangGraph-based agent
from src.graph.f150_graph import create_f150_graph
from src.utils.approval_node import format_approval_prompt_for_cli
from src.config import Config
from src.rag import create_vector_store


def print_startup_banner():
    """Print the startup initialization banner."""
    print("=" * 70)
    print("F150 EXPERT AGENT (LangGraph) - INITIALIZING")
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
    print("2018 Ford F-150 Expert Assistant (LangGraph + Memory)")
    print("=" * 70)
    print("Ask me anything about your 2018 F-150!")
    print("   Your conversation history is preserved across messages!")
    if Config.TOOL_APPROVAL_ENABLED:
        print("   [HITL Mode: You will be asked to approve tool calls]")
    print("\nType 'quit' to exit")
    print("=" * 70)

def print_session_summary(agent):
    """Print the session summary when exiting."""
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)

    # Get the final state from the checkpointer
    try:
        state = agent.get_state({"configurable": {"thread_id": "1"}})
        total_tokens = state.values.get("total_tokens", 0)
        total_prompt_tokens = state.values.get("total_prompt_tokens", 0)
        total_completion_tokens = state.values.get("total_completion_tokens", 0)
        context_limit = state.values.get("context_limit", 128000)

        usage_percentage = (total_tokens / context_limit) * 100 if context_limit > 0 else 0
        remaining_tokens = context_limit - total_tokens

        # Count interactions by counting user messages
        messages = state.values.get("messages", [])
        interactions = sum(1 for m in messages if hasattr(m, 'type') and m.type == 'human')

        print(f"Total tokens used: {total_tokens:,}")
        print(f"  Prompt tokens: {total_prompt_tokens:,}")
        print(f"  Completion tokens: {total_completion_tokens:,}")
        print(f"Total interactions: {interactions}")
        print(f"Final context usage: {usage_percentage:.1f}%")
        print(f"Remaining tokens: {remaining_tokens:,}")
    except Exception as e:
        print(f"Unable to retrieve session summary: {e}")

    print("=" * 70)

def read_input(agent):
    """
    Read user input and handle quit commands.

    Args:
        agent: The LangGraph agent for retrieving session summary

    Returns:
        User input string, or None if user wants to quit
    """
    user_input = input("\nYou: ").strip()

    if user_input.lower() in ["quit", "exit", "q"]:
        print_session_summary(agent)
        print("\nGoodbye!")
        return None

    return user_input


def main():
    """Run the F150 expert agent in interactive mode with LangGraph and memory."""
    # Validate configuration
    if not Config.validate():
        return

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

    print("\nStep 2: Initializing LangGraph agent with conversation memory...")

    # Create the LangGraph agent with the vector store and checkpointer
    try:
        agent = create_f150_graph(vector_store=vector_store)
        print("✓ LangGraph agent ready!")
        if Config.TOKEN_TRACKING_ENABLED:
            print("✓ Token tracking enabled (integrated in graph)")
    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        return

    # Interactive loop
    print_welcome_message()

    while True:
        user_input = read_input(agent)

        if user_input is None:
            break

        if not user_input:
            continue

        try:
            # Invoke LangGraph agent with conversation memory
            config = {"configurable": {"thread_id": "1"}}
            response = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config
            )

            # Check if execution was interrupted for approval
            while response.get("__interrupt__"):
                interrupt_data = response["__interrupt__"][0].value

                # Format and display the approval request
                approval_prompt = format_approval_prompt_for_cli(interrupt_data)
                print(approval_prompt, end="")

                # Get user's approval decision
                approval_input = input().strip().lower()

                if approval_input in ['y', 'yes']:
                    # Approve - resume with True
                    response = agent.invoke(
                        Command(resume={"approved": True}),
                        config
                    )
                elif approval_input in ['n', 'no']:
                    # Reject - resume with False
                    response = agent.invoke(
                        Command(resume={"approved": False}),
                        config
                    )
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
                    # Re-prompt (don't invoke, just loop to ask again)
                    continue

            # Extract and display the final message
            final_message_obj = response["messages"][-1]
            print(f"\nF150 Expert (LangGraph):\n{final_message_obj.content}")

            # Token tracking is now handled automatically by the token_tracker node

        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
