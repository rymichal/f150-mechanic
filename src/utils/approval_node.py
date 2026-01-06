"""
Human-in-the-loop approval node for tool execution.

This module provides a node that intercepts tool calls and uses LangGraph's
interrupt() function to pause execution for human approval before tools execute.
"""

from typing import Dict
from langgraph.types import interrupt
from langchain_core.messages import AIMessage
from src.graph.state import F150StateWithDualContext
from src.config import Config


def create_approval_node(enabled: bool = True):
    """
    Factory function to create a tool approval node for human-in-the-loop.

    This node intercepts tool calls from the agent and pauses execution
    using LangGraph's interrupt() to request human approval. The approval
    decision determines whether tools execute or the agent responds without them.

    Args:
        enabled: Whether to enable approval (default: True). If False,
                 node passes through without interrupting.

    Returns:
        A node function that can be added to a LangGraph workflow

    Usage:
        >>> workflow.add_node("approval_gate", create_approval_node(enabled=True))
    """

    def approval_node(state: F150StateWithDualContext) -> Dict:
        """
        Approval gate node that requests human approval for tool execution.

        If approval is enabled and tool calls are present:
        1. Extract tool call information from the last AI message
        2. Call interrupt() with tool details (pauses execution)
        3. Resume returns approval decision (True/False)
        4. If rejected, inject a message telling agent to respond without tools

        Args:
            state: Current graph state with messages

        Returns:
            Dict with optional messages (if tools rejected)
        """
        # If approval is disabled, pass through
        if not enabled:
            return {}

        if Config.TELEMETRY:
            print("\n✋ APPROVAL_GATE: Requesting human approval for tool execution...")

        messages = state["messages"]
        last_message = messages[-1]

        # Check if there are tool calls to approve
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            # No tool calls, nothing to approve
            return {}

        # Build approval prompt with tool details
        tool_calls = last_message.tool_calls
        if Config.TELEMETRY:
            tool_names = [tc.get('name') for tc in tool_calls]
            print(f"  Tools requested: {tool_names}")

        approval_prompt = _build_approval_prompt(tool_calls)

        # Interrupt execution and wait for human decision
        # This pauses the graph and returns control to the caller
        # The resume value will be the approval decision (True/False or dict)
        approval_decision = interrupt(approval_prompt)

        # Handle the approval decision
        if approval_decision is False or (isinstance(approval_decision, dict) and approval_decision.get("approved") is False):
            # Tools rejected - inject message telling agent to respond without tools
            if Config.TELEMETRY:
                print("  ✗ Tools rejected - agent will respond without tools")
            rejection_message = AIMessage(
                content="[SYSTEM: The requested tool calls were not approved. Please respond to the user's question directly without using tools.]",
                tool_calls=[]  # Clear tool calls
            )
            return {"messages": [rejection_message]}

        # Tools approved - continue to tool execution
        if Config.TELEMETRY:
            print("  ✓ Tools approved - proceeding to execution")
        return {}

    return approval_node


def _build_approval_prompt(tool_calls: list) -> Dict:
    """
    Build a human-readable approval prompt from tool calls.

    Args:
        tool_calls: List of tool call objects from the AI message

    Returns:
        Dict with approval request details in a structured format
    """
    tools_info = []
    for tool_call in tool_calls:
        tool_info = {
            "name": tool_call.get("name", "unknown"),
            "args": tool_call.get("args", {}),
            "id": tool_call.get("id", "")
        }
        tools_info.append(tool_info)

    return {
        "type": "tool_approval_request",
        "tools": tools_info,
        "message": f"Approve execution of {len(tool_calls)} tool(s)?"
    }


def format_approval_prompt_for_cli(interrupt_data: Dict) -> str:
    """
    Format the interrupt data into a CLI-friendly prompt.

    This is a helper function for the main script to display approval
    requests to the user in a readable format.

    Args:
        interrupt_data: The interrupt payload from the approval node

    Returns:
        Formatted string for CLI display

    Usage:
        >>> prompt = format_approval_prompt_for_cli(result["__interrupt__"][0].value)
        >>> print(prompt)
    """
    if not isinstance(interrupt_data, dict) or interrupt_data.get("type") != "tool_approval_request":
        return str(interrupt_data)

    tools = interrupt_data.get("tools", [])
    lines = [
        "\n" + "=" * 70,
        "TOOL APPROVAL REQUEST",
        "=" * 70,
    ]

    for i, tool in enumerate(tools, 1):
        lines.append(f"\n[{i}] Tool: {tool.get('name', 'unknown')}")
        lines.append(f"    Arguments:")
        args = tool.get('args', {})
        for key, value in args.items():
            # Truncate long values for display
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            lines.append(f"      {key}: {value_str}")

    lines.append("\n" + "=" * 70)
    lines.append(f"Approve execution of these {len(tools)} tool(s)? (y/n): ")

    return "\n".join(lines)
