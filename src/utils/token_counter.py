"""Token counting utilities for tracking context usage with Ollama."""

from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage


class OllamaTokenCounter:
    """Utility class for tracking token usage from Ollama responses."""

    def __init__(self, context_limit: int = 128000):
        """
        Initialize the Ollama token counter.

        Args:
            context_limit: Maximum context window size in tokens (default: 128k for llama3.2)
        """
        self.context_limit = context_limit
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.message_history: List[Dict[str, Any]] = []

    def extract_token_counts(self, message: BaseMessage) -> Optional[Dict[str, int]]:
        """
        Extract token counts from an Ollama response message.

        Args:
            message: LangChain AIMessage with response_metadata from Ollama

        Returns:
            Dictionary with token counts if available, None otherwise
        """
        if not hasattr(message, 'response_metadata'):
            return None

        metadata = message.response_metadata

        # Ollama returns prompt_eval_count and eval_count
        prompt_tokens = metadata.get('prompt_eval_count')
        completion_tokens = metadata.get('eval_count')

        if prompt_tokens is None or completion_tokens is None:
            return None

        return {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens
        }

    def track_interaction(
        self,
        prompt_tokens: int,
        completion_tokens: int
    ) -> Dict[str, Any]:
        """
        Track a single interaction and update cumulative token count.

        Args:
            prompt_tokens: Number of tokens in the prompt (from Ollama)
            completion_tokens: Number of tokens in the completion (from Ollama)

        Returns:
            Dictionary with token usage statistics
        """
        interaction_tokens = prompt_tokens + completion_tokens

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += interaction_tokens

        usage_stats = {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'interaction_tokens': interaction_tokens,
            'cumulative_prompt_tokens': self.total_prompt_tokens,
            'cumulative_completion_tokens': self.total_completion_tokens,
            'cumulative_tokens': self.total_tokens,
            'context_limit': self.context_limit,
            'usage_percentage': (self.total_tokens / self.context_limit) * 100,
            'remaining_tokens': self.context_limit - self.total_tokens
        }

        self.message_history.append(usage_stats)
        return usage_stats

    def get_context_percentage(self) -> float:
        """
        Get the current context usage as a percentage.

        Returns:
            Percentage of context used (0-100)
        """
        return (self.total_tokens / self.context_limit) * 100

    def get_remaining_tokens(self) -> int:
        """
        Get the number of remaining tokens in the context window.

        Returns:
            Number of tokens remaining
        """
        return max(0, self.context_limit - self.total_tokens)

    def is_near_limit(self, threshold: float = 80.0) -> bool:
        """
        Check if context usage is approaching the limit.

        Args:
            threshold: Percentage threshold to consider "near limit" (default: 80%)

        Returns:
            True if usage exceeds threshold, False otherwise
        """
        return self.get_context_percentage() >= threshold

    def reset(self) -> None:
        """Reset the token counter and message history."""
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.message_history = []

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of token usage.

        Returns:
            Dictionary with usage summary statistics
        """
        return {
            'total_tokens': self.total_tokens,
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'context_limit': self.context_limit,
            'usage_percentage': self.get_context_percentage(),
            'remaining_tokens': self.get_remaining_tokens(),
            'total_interactions': len(self.message_history),
            'is_near_limit': self.is_near_limit()
        }


def format_token_usage(usage_stats: Dict[str, Any], show_details: bool = True) -> str:
    """
    Format token usage statistics for display.

    Args:
        usage_stats: Dictionary containing token usage statistics
        show_details: Whether to show detailed breakdown (default: True)

    Returns:
        Formatted string for display
    """
    lines = []

    if show_details:
        lines.append(f"  Prompt: {usage_stats['prompt_tokens']:,} tokens")
        lines.append(f"  Completion: {usage_stats['completion_tokens']:,} tokens")
        lines.append(f"  This interaction: {usage_stats['interaction_tokens']:,} tokens")

    lines.append(f"  Cumulative: {usage_stats['cumulative_tokens']:,} / {usage_stats['context_limit']:,} tokens ({usage_stats['usage_percentage']:.1f}%)")
    lines.append(f"  Remaining: {usage_stats['remaining_tokens']:,} tokens")

    return "\n".join(lines)


def get_progress_bar(percentage: float, width: int = 40) -> str:
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
        bar_char = '█'
        label = 'LOW'
    elif percentage < 80:
        bar_char = '█'
        label = 'MEDIUM'
    else:
        bar_char = '█'
        label = 'HIGH'

    bar = f"[{bar_char * filled}{'-' * empty}] {percentage:.1f}% ({label})"
    return bar


def get_warning_message(usage_stats: Dict[str, Any]) -> Optional[str]:
    """
    Get a warning message if context usage is high.

    Args:
        usage_stats: Dictionary containing token usage statistics

    Returns:
        Warning message if needed, None otherwise
    """
    percentage = usage_stats['usage_percentage']

    if percentage >= 95:
        return "⚠️  CRITICAL: Context nearly full! Consider starting a new conversation."
    elif percentage >= 80:
        return "⚠️  WARNING: Context usage is high. Approaching limit."
    elif percentage >= 60:
        return "ℹ️  Context usage is moderate."

    return None
