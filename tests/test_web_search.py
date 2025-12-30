"""Test script for web search using Brave Search API."""

import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from src.tools.web_search import search_web
from src.config import Config


def main():
    print("=" * 70)
    print("Testing Brave Web Search")
    print("=" * 70)
    print()

    # Check if API key is configured
    if not Config.BRAVE_API_KEY or Config.BRAVE_API_KEY == "your_brave_api_key_here":
        print("❌ ERROR: BRAVE_API_KEY not configured")
        print("\nTo fix this:")
        print("  1. Get a free API key from https://brave.com/search/api/")
        print("  2. Add it to your .env file:")
        print("     BRAVE_API_KEY=your_actual_api_key")
        return

    # Test query
    test_query = "2018 Ford F150 door latch recall"

    print(f"Test Query: '{test_query}'")
    print("=" * 70)
    print()

    try:
        # Call the search_web tool (it's a StructuredTool, so we use .invoke())
        results = search_web.invoke({"query": test_query})

        print("Results:")
        print(results)

        print("\n" + "=" * 70)
        print("SUCCESS: Web search is working!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nMake sure:")
        print("  1. Your BRAVE_API_KEY is valid")
        print("  2. You have internet connectivity")
        print("  3. The Brave Search API service is accessible")


if __name__ == "__main__":
    main()
