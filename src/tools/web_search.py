"""
Generic web search tool using Brave Search API.

This tool enables agents to search the web for current information,
troubleshooting tips, and real-world knowledge not found in static documents.
"""

from langchain.tools import tool
from langchain_community.tools import BraveSearch
from ..config import Config


def _get_brave_search():
    """Create and configure Brave Search tool."""
    if not Config.BRAVE_API_KEY:
        raise ValueError("BRAVE_API_KEY not found in environment variables")

    return BraveSearch.from_api_key(
        api_key=Config.BRAVE_API_KEY,
        search_kwargs={"count": 5}  # Return top 5 results
    )


@tool
def search_web(query: str) -> str:
    """
    Search the web for current information and real-world knowledge.

    Use this tool to find:
    - Current events and recent information
    - Troubleshooting tips and solutions
    - Product recalls and known issues
    - User experiences and reviews
    - Technical specifications not in manuals
    - Forum discussions and community wisdom

    Args:
        query: The search query

    Returns:
        Web search results with summaries and links

    Example:
        >>> search_web("frozen door latch problem")
        "Web search results for 'frozen door latch problem'..."
    """

    print(f"\n🌐 Searching web for: '{query}'...")

    try:
        brave_search = _get_brave_search()
        results = brave_search.run(query)

        if not results or results.strip() == "":
            return f"No web results found for: {query}"

        return results

    except ValueError as e:
        return f"Web search not configured: {str(e)}. Please add BRAVE_API_KEY to your .env file."
    except Exception as e:
        return f"Error searching web: {str(e)}"
