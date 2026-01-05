"""
Token tracking node for LangGraph agents.

This module provides a node that tracks token usage from Ollama responses
and integrates it into the LangGraph state, allowing for reactive behavior
based on context usage.
"""

from typing import Dict
from langchain_core.messages import AIMessage, SystemMessage
from src.graph.state import F150StateWithTokens


def create_token_tracking_node(context_limit: int = 128000, warning_threshold: float = 80.0):
    """
    Factory function to create a token tracking node for LangGraph.

    This creates a node that:
    - Extracts token counts from Ollama response metadata
    - Updates cumulative token counts in graph state
    - Calculates usage percentage and remaining tokens
    - Optionally injects warning messages into the conversation
    - Displays token usage to the console

    Args:
        context_limit: Maximum context window size in tokens (default: 128k)
        warning_threshold: Percentage threshold to trigger warnings (default: 80%)

    Returns:
        A node function that can be added to a LangGraph workflow

    Usage:
        >>> workflow.add_node("token_tracker", create_token_tracking_node(128000, 80.0))
    """

    def token_tracking_node(state: F150StateWithTokens) -> Dict:
        """
        Token tracking node that processes the last AI message.

        This node extracts token usage from Ollama's response metadata,
        updates the cumulative totals in state, and optionally injects
        warning messages if context usage is high.

        Args:
            state: Current graph state with messages and token tracking fields

        Returns:
            Dictionary with updated token counts and optional warning message
        """
        messages = state["messages"]

        # Find the last AI message with response metadata
        last_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and hasattr(msg, 'response_metadata'):
                last_ai_message = msg
                break

        if not last_ai_message:
            return {}  # No AI message to track

        # Extract token counts from Ollama metadata
        metadata = last_ai_message.response_metadata
        prompt_tokens = metadata.get('prompt_eval_count', 0)
        completion_tokens = metadata.get('eval_count', 0)

        if prompt_tokens == 0 and completion_tokens == 0:
            # No token data available from Ollama
            print("\n⚠️  Token tracking unavailable - Ollama did not return token counts")
            return {}

        # Update cumulative totals
        new_total_prompt = state.get("total_prompt_tokens", 0) + prompt_tokens
        new_total_completion = state.get("total_completion_tokens", 0) + completion_tokens
        new_total_tokens = new_total_prompt + new_total_completion

        # Calculate usage metrics
        usage_percentage = (new_total_tokens / context_limit) * 100
        remaining_tokens = context_limit - new_total_tokens

        # Build state updates
        updates = {
            "total_prompt_tokens": new_total_prompt,
            "total_completion_tokens": new_total_completion,
            "total_tokens": new_total_tokens,
            "context_limit": context_limit,
        }

        # Display token usage to console
        _display_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            interaction_tokens=prompt_tokens + completion_tokens,
            cumulative_tokens=new_total_tokens,
            context_limit=context_limit,
            usage_percentage=usage_percentage,
            remaining_tokens=remaining_tokens
        )

        # Optional: Inject warning message if usage is high
        warning_message = _get_warning_message(usage_percentage, remaining_tokens)
        if warning_message:
            updates["messages"] = [SystemMessage(content=warning_message)]

        return updates

    return token_tracking_node


def _display_token_usage(
    prompt_tokens: int,
    completion_tokens: int,
    interaction_tokens: int,
    cumulative_tokens: int,
    context_limit: int,
    usage_percentage: float,
    remaining_tokens: int
) -> None:
    """
    Display token usage statistics to the console.

    Args:
        prompt_tokens: Tokens used in the prompt
        completion_tokens: Tokens used in the completion
        interaction_tokens: Total tokens for this interaction
        cumulative_tokens: Total tokens used across all interactions
        context_limit: Maximum context window size
        usage_percentage: Percentage of context used
        remaining_tokens: Tokens remaining in context
    """
    print("\n" + "-" * 70)
    print("TOKEN USAGE (Ollama actual):")
    print(f"  Prompt: {prompt_tokens:,} tokens")
    print(f"  Completion: {completion_tokens:,} tokens")
    print(f"  This interaction: {interaction_tokens:,} tokens")
    print(f"  Cumulative: {cumulative_tokens:,} / {context_limit:,} tokens ({usage_percentage:.1f}%)")
    print(f"  Remaining: {remaining_tokens:,} tokens")

    # Display progress bar
    progress_bar = _get_progress_bar(usage_percentage)
    print(f"\nContext: {progress_bar}")

    print("-" * 70)


def _get_progress_bar(percentage: float, width: int = 40) -> str:
    """
    Create a visual progress bar for context usage.

    Args:
        percentage: Usage percentage (0-100)
        width: Width of the progress bar in characters (default: 40)

    Returns:
        String representation of a progress bar
    """
    filled = int((percentage / 100) * width)
    empty = width - filled

    # Color coding based on usage
    if percentage < 50:
        label = 'LOW'
    elif percentage < 80:
        label = 'MEDIUM'
    else:
        label = 'HIGH'

    bar = f"[{'█' * filled}{'-' * empty}] {percentage:.1f}% ({label})"
    return bar


def _get_warning_message(usage_percentage: float, remaining_tokens: int) -> str | None:
    """
    Get a warning message if context usage is high.

    Args:
        usage_percentage: Current usage percentage (0-100)
        remaining_tokens: Number of tokens remaining

    Returns:
        Warning message string if needed, None otherwise
    """
    if usage_percentage >= 95:
        return f"⚠️ CRITICAL: Context nearly full ({usage_percentage:.1f}%, {remaining_tokens:,} tokens remaining). Consider starting a new conversation."
    elif usage_percentage >= 80:
        return f"⚠️ WARNING: Context usage is high ({usage_percentage:.1f}%, {remaining_tokens:,} tokens remaining)."
    elif usage_percentage >= 60:
        return f"ℹ️ Context usage is moderate ({usage_percentage:.1f}%, {remaining_tokens:,} tokens remaining)."

    return None
