"""Test script for Ollama token counter functionality."""

from src.utils.token_counter import (
    OllamaTokenCounter,
    format_token_usage,
    get_progress_bar,
    get_warning_message
)
from langchain_core.messages import AIMessage


def test_basic_tracking():
    """Test basic token counting and tracking."""
    print("=" * 70)
    print("TEST 1: Basic Token Tracking")
    print("=" * 70)

    counter = OllamaTokenCounter(context_limit=1000)

    # Simulate interactions with actual token counts
    stats1 = counter.track_interaction(
        prompt_tokens=50,
        completion_tokens=100
    )

    print("\nInteraction 1:")
    print(format_token_usage(stats1))
    print(f"\nProgress: {get_progress_bar(stats1['usage_percentage'])}")

    # Second interaction
    stats2 = counter.track_interaction(
        prompt_tokens=75,
        completion_tokens=125
    )

    print("\n\nInteraction 2:")
    print(format_token_usage(stats2))
    print(f"\nProgress: {get_progress_bar(stats2['usage_percentage'])}")

    print("\n✓ Test 1 passed\n")


def test_extract_token_counts():
    """Test extraction of token counts from AIMessage."""
    print("=" * 70)
    print("TEST 2: Extract Token Counts from AIMessage")
    print("=" * 70)

    counter = OllamaTokenCounter()

    # Simulate an Ollama response with metadata
    message_with_tokens = AIMessage(
        content="This is a test response",
        response_metadata={
            'model': 'llama3.2',
            'prompt_eval_count': 245,
            'eval_count': 892,
            'done': True
        }
    )

    token_counts = counter.extract_token_counts(message_with_tokens)

    print(f"\nExtracted token counts: {token_counts}")
    assert token_counts is not None, "Should extract token counts"
    assert token_counts['prompt_tokens'] == 245
    assert token_counts['completion_tokens'] == 892
    assert token_counts['total_tokens'] == 1137

    # Test message without metadata
    message_without_tokens = AIMessage(content="No metadata")
    no_counts = counter.extract_token_counts(message_without_tokens)

    print(f"Message without metadata: {no_counts}")
    assert no_counts is None, "Should return None when no metadata"

    print("\n✓ Test 2 passed\n")


def test_warnings():
    """Test warning messages at different usage levels."""
    print("=" * 70)
    print("TEST 3: Warning Messages")
    print("=" * 70)

    counter = OllamaTokenCounter(context_limit=100)

    # Test different usage levels
    test_cases = [
        (20, 30, "Low usage"),
        (30, 40, "Moderate usage"),
        (40, 45, "High usage"),
    ]

    for prompt, completion, description in test_cases:
        stats = counter.track_interaction(prompt, completion)
        warning = get_warning_message(stats)

        print(f"\n{description} ({stats['usage_percentage']:.1f}%):")
        if warning:
            print(f"  {warning}")
        else:
            print("  No warning")

    print("\n✓ Test 3 passed\n")


def test_summary():
    """Test session summary."""
    print("=" * 70)
    print("TEST 4: Session Summary")
    print("=" * 70)

    counter = OllamaTokenCounter(context_limit=128000)

    # Simulate a conversation
    interactions = [
        (50, 100),
        (60, 120),
        (70, 130),
    ]

    for prompt_tokens, completion_tokens in interactions:
        counter.track_interaction(prompt_tokens, completion_tokens)

    summary = counter.get_summary()

    print(f"\nTotal tokens: {summary['total_tokens']:,}")
    print(f"Total prompt tokens: {summary['total_prompt_tokens']:,}")
    print(f"Total completion tokens: {summary['total_completion_tokens']:,}")
    print(f"Context limit: {summary['context_limit']:,}")
    print(f"Usage: {summary['usage_percentage']:.1f}%")
    print(f"Remaining: {summary['remaining_tokens']:,}")
    print(f"Total interactions: {summary['total_interactions']}")
    print(f"Near limit: {summary['is_near_limit']}")

    # Verify calculations
    assert summary['total_tokens'] == 530  # 50+100+60+120+70+130
    assert summary['total_prompt_tokens'] == 180  # 50+60+70
    assert summary['total_completion_tokens'] == 350  # 100+120+130
    assert summary['total_interactions'] == 3

    print("\n✓ Test 4 passed\n")


def test_progress_bar():
    """Test progress bar visualization."""
    print("=" * 70)
    print("TEST 5: Progress Bar Visualization")
    print("=" * 70)

    usage_levels = [0, 25, 50, 75, 85, 95, 100]

    for usage in usage_levels:
        bar = get_progress_bar(usage)
        print(f"\n{usage:3d}%: {bar}")

    print("\n✓ Test 5 passed\n")


def main():
    """Run all tests."""
    print("\nOLLAMA TOKEN COUNTER TEST SUITE")
    print("=" * 70)

    try:
        test_basic_tracking()
        test_extract_token_counts()
        test_warnings()
        test_summary()
        test_progress_bar()

        print("=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
