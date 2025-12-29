from langchain.tools import tool

from ..config import Config


@tool
def get_current_location() -> str:
    """Get the user's current location. Returns Grand Rapids, Michigan as the default location."""
    return Config.DEFAULT_LOCATION
