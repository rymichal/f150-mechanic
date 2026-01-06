"""
Conversational message filter for LangGraph agents.

This module provides utilities to detect and handle purely conversational messages
(greetings, thanks, acknowledgments) without invoking LLM tools, improving performance
and reducing unnecessary tool calls.
"""

import re
from typing import Dict
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage
from src.config import Config


def is_conversational_only(text: str) -> bool:
    """
    Detect if a message is purely conversational (no domain-specific question).

    This function uses regex patterns to identify common conversational phrases
    that don't require tool usage or complex LLM reasoning.

    Args:
        text: The user's message text

    Returns:
        True if the message is conversational-only, False otherwise

    Examples:
        >>> is_conversational_only("thank you")
        True
        >>> is_conversational_only("What is the oil capacity?")
        False
        >>> is_conversational_only("Great!")
        True
    """
    text_lower = text.lower().strip()

    # Patterns for conversational-only messages
    conversational_patterns = [
        # Greetings
        r'^(hi|hello|hey|sup|yo|howdy)[\s!.]*$',

        # Thanks
        r'^(thank you|thanks|thx|ty|thank u|tysm|appreciate it)[\s!.]*$',

        # Acknowledgments
        r'^(great|ok|okay|got it|cool|nice|perfect|awesome|excellent)[\s!.]*$',

        # Combined acknowledgment + thanks
        r'^(great|ok|okay|cool|nice|perfect|awesome)\s*(thank you|thanks|thx)[\s!.]*$',

        # Farewells
        r'^(bye|goodbye|see you|later|cya|take care)[\s!.]*$',

        # Affirmations
        r'^(yes|yeah|yep|yup|sure|alright)[\s!.]*$',
        r'^(no|nope|nah)[\s!.]*$',
    ]

    for pattern in conversational_patterns:
        if re.match(pattern, text_lower):
            return True

    return False


def get_conversational_response(text: str) -> str:
    """
    Generate an appropriate canned response for conversational messages.

    Args:
        text: The user's conversational message

    Returns:
        An appropriate response string
    """
    text_lower = text.lower().strip()

    # Response mapping based on message type
    response_map = {
        'thank': "You're welcome! Let me know if you have any other questions about your F-150!",
        'great': "Glad I could help! Feel free to ask anything else about your 2018 F-150.",
        'ok': "Great! Let me know if there's anything else I can help with.",
        'hi': "Hello! How can I help you with your 2018 F-150 today?",
        'hello': "Hello! How can I help you with your 2018 F-150 today?",
        'hey': "Hey there! What can I help you with regarding your F-150?",
        'bye': "Goodbye! Come back anytime you have F-150 questions!",
        'yes': "Got it! Anything else you'd like to know?",
        'no': "No problem! Let me know if you need anything.",
    }

    # Find matching keyword and return response
    for keyword, response in response_map.items():
        if keyword in text_lower:
            return response

    # Default fallback response
    return "You're welcome! Feel free to ask if you have any questions about your 2018 F-150."


def create_conversational_filter_node(domain_name: str = "F-150"):
    """
    Factory function to create a pre-filter node for conversational messages.

    This node intercepts purely conversational messages and returns canned responses
    without invoking the LLM with tools, reducing latency and token usage.

    Args:
        domain_name: The domain/topic name to use in responses (e.g., "F-150")

    Returns:
        A node function that can be added to a LangGraph workflow

    Usage:
        >>> workflow.add_node("pre_filter", create_conversational_filter_node("F-150"))
    """
    def pre_filter_node(state: MessagesState) -> Dict:
        """
        Pre-filter node that checks if the message is conversational-only.

        If the message is conversational, returns a direct response and sets
        bypass_agent=True to skip the agent node.

        Args:
            state: The current graph state containing messages

        Returns:
            Dict with messages and bypass_agent flag
        """
        if Config.TELEMETRY:
            print("\n🔍 PRE_FILTER: Checking if message is conversational...")

        messages = state["messages"]

        # Get the last user message
        last_user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg
                break

        # Check if it's conversational-only
        if last_user_message and is_conversational_only(last_user_message.content):
            if Config.TELEMETRY:
                print("  ✓ Conversational message detected - bypassing agent")
            response_text = get_conversational_response(last_user_message.content)

            # Customize response with domain name
            if domain_name and domain_name != "F-150":
                response_text = response_text.replace("F-150", domain_name).replace("2018 F-150", domain_name)

            return {
                "messages": [AIMessage(content=response_text)],
                "bypass_agent": True
            }

        # Not conversational-only, proceed to agent
        if Config.TELEMETRY:
            print("  ✓ Domain question detected - proceeding to agent")
        return {"bypass_agent": False}

    return pre_filter_node
