import warnings

# Suppress Pydantic v1 compatibility warning in Python 3.14
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from langchain_core.messages import HumanMessage

from src.agent import create_weather_agent
from src.config import Config


def main():
    """Run the weather agent in interactive mode."""
    # Validate configuration
    if not Config.validate():
        return

    # Create the agent
    agent = create_weather_agent()

    # Interactive loop
    print("Weather Agent - Type 'quit' to exit")
    print("=" * 50)

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
            print(f"\nAgent: {final_message}")
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
