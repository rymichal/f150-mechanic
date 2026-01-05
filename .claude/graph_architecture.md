# F150 Agent Graph Architecture

## Graph Flow Diagram

```
START
  │
  ▼
┌─────────────┐
│ pre_filter  │ ◄── Intercepts conversational-only messages
└─────────────┘     (greetings, thanks, acknowledgments)
  │
  ├─── [conversational] ──► END (bypass agent & token tracking)
  │
  └─── [needs agent] ──►┐
                        ▼
                   ┌────────┐
                   │ agent  │ ◄── LLM decides: respond or use tools
                   └────────┘
                        │
                        ├─── [has tool_calls] ──►┐
                        │                        ▼
                        │                   ┌────────┐
                        │                   │ tools  │ ◄── Execute tool calls
                        │                   └────────┘
                        │                        │
                        │                        └─── (loop back to agent)
                        │
                        └─── [no tool_calls] ──►┐
                                                 ▼
                                          ┌──────────────────┐
                                          │ token_tracker    │ ◄── Track token usage
                                          └──────────────────┘
                                                 │
                                                 ▼
                                               END
```

## Node Descriptions

### 1. **pre_filter** (Conversational Filter)
- **Purpose**: Intercept purely conversational messages and respond immediately
- **Input**: User message
- **Output**: Direct response OR pass-through to agent
- **Benefit**: Eliminates unnecessary LLM calls for simple pleasantries

### 2. **agent** (LLM Reasoning Node)
- **Purpose**: Call the LLM with tools to decide next action
- **Input**: User message + conversation history
- **Output**: AI response with optional tool calls
- **Tools Available**: `search_f150_manual`, `search_web`

### 3. **tools** (Tool Execution Node)
- **Purpose**: Execute tool calls made by the agent
- **Input**: Tool calls from agent
- **Output**: Tool results
- **Behavior**: Automatically loops back to agent with results

### 4. **token_tracker** (Token Tracking Node)
- **Purpose**: Track token usage and inject warnings if needed
- **Input**: Last AI message with response metadata
- **Output**: Updated token counts in state + optional warning message
- **Benefit**:
  - Tracks context usage across conversation
  - Persists in checkpointer (survives restarts)
  - Can inject warnings when approaching limit

## State Schema

### F150StateWithTokens
Extends `MessagesState` with token tracking fields:

```python
{
    "messages": [...],              # Conversation messages (from MessagesState)
    "total_tokens": 0,              # Cumulative total tokens
    "total_prompt_tokens": 0,       # Cumulative prompt tokens
    "total_completion_tokens": 0,   # Cumulative completion tokens
    "context_limit": 128000,        # Max context window (128k for llama3.2)
    "bypass_agent": False           # Flag for pre_filter
}
```

## Benefits of Token Tracking as a Node

1. **Persistence**: Token counts are saved in the checkpointer (SQLite)
   - Survive app restarts
   - Per-conversation tracking

2. **Reactivity**: Can inject warning messages into the conversation
   - 60%: "ℹ️ Context usage is moderate"
   - 80%: "⚠️ WARNING: Context usage is high"
   - 95%: "⚠️ CRITICAL: Context nearly full"

3. **Integration**: Part of the graph execution flow
   - Automatic tracking on every response
   - No manual tracking needed in main loop
   - Consistent with LangGraph architecture

4. **Future Extensions**: Easy to add features like:
   - Auto-summarization when context is high
   - Smart message pruning
   - Context-aware responses

## Execution Flow Examples

### Example 1: Conversational Message
```
User: "Great thank you"
  → pre_filter (detects conversational)
  → END (Response: "You're welcome! Let me know if you have any other questions...")
```
**Nodes executed**: 1 (pre_filter only)

### Example 2: Simple Question (No Tools)
```
User: "How are you?"
  → pre_filter (not conversational, passes through)
  → agent (LLM responds without tools)
  → token_tracker (tracks usage)
  → END
```
**Nodes executed**: 3 (pre_filter → agent → token_tracker)

### Example 3: Question Requiring Tools
```
User: "What is the oil capacity?"
  → pre_filter (passes through)
  → agent (decides to use search_f150_manual)
  → tools (searches manual)
  → agent (formulates response with results)
  → token_tracker (tracks usage)
  → END
```
**Nodes executed**: 5 (pre_filter → agent → tools → agent → token_tracker)

## Configuration

Token tracking is controlled via `Config.TOKEN_TRACKING_ENABLED`:
- **Enabled**: Token tracker node runs and displays stats
- **Disabled**: Token tracker still runs but minimal output

Context limit is set via `Config.CONTEXT_LIMIT` (default: 128000 for llama3.2)
