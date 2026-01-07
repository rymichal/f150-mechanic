### Running Python

```bash
# LangChain version (no persistent memory)
uv run python main_chain.py

# LangGraph version (with persistent memory)
uv run python main.py

# Check Syntax with python
uv run python -m py_compile src/agent/f150_agent.py
```

# Role Prompt

You are an expert AI engineering building a langChain/langGraph application. Its primary object is to explore the uses and functionality of the technical stack and working with AI agents. The secondary objective is to build a AI agent that can assist users with using their ford f150 truck.

# LangChain Tutorial - Project Context

This repository contains multiple AI agent projects demonstrating different approaches to building LLM-powered applications using LangChain and LangGraph with Ollama.

**Package Management**: This project uses `uv` for Python package management and versioning. Use `uv` commands for installing dependencies and managing the Python environment.

## Projects in this Repository

### 1. Weather Agent (Simple Example)
- **Entry Point**: Not currently active (example/reference implementation)
- **Location**: `src/agent/weather_agent.py`
- **Description**: Basic weather agent demonstrating LangChain tool usage
- **Technologies**: LangChain, simple tool calling pattern

### 2. F150 Expert Agent - LangChain (High-Level API)
- **Entry Point**: `main.py`
- **Agent Code**: `src/agent/f150_agent.py`
- **Description**: 2018 Ford F-150 expert assistant using LangChain's `create_agent()` high-level API
- **Technologies**:
  - LangChain `create_agent` with `InMemorySaver` checkpointer
  - RAG (Retrieval-Augmented Generation) using FAISS vector store
  - Ollama with llama3.2 model
  - Two tools: Manual search (RAG) and Web search (Brave API)
- **Memory**: In-memory conversation memory (temporary, session-based)
- **Use Case**: Quick agent setup with minimal code

### 3. F150 Expert Agent - LangGraph (Low-Level Orchestration)
- **Entry Point**: `main_graph.py`
- **Graph Code**: `src/graph/f150_graph.py`
- **Description**: Same F150 expert functionality but implemented with LangGraph primitives for full control
- **Technologies**:
  - LangGraph `StateGraph` with explicit nodes, edges, and state management
  - Same RAG and tools as LangChain version
  - SqliteSaver checkpointer for persistent conversation memory
- **Memory**: Persistent SQLite-based conversation memory (survives across sessions)
- **Use Case**: Custom orchestration, fine-grained control, production-ready memory

## Project Structure

```
langchain-tutorial/
├── main.py                          # Entry point for LangChain F150 agent
├── main_graph.py                    # Entry point for LangGraph F150 agent
├── conversations.db                 # SQLite database for conversation memory (auto-generated)
│
├── src/
│   ├── agent/                       # LangChain agents (high-level)
│   │   ├── f150_agent.py           # F150 agent using create_agent()
│   │   └── weather_agent.py        # Simple weather agent example
│   │
│   ├── graph/                       # LangGraph implementations (low-level)
│   │   └── f150_graph.py           # F150 agent using StateGraph, nodes, edges
│   │
│   ├── rag/                         # RAG (Retrieval-Augmented Generation) components
│   │   ├── document_loader.py      # PDF document loading and chunking
│   │   ├── embeddings.py           # Ollama embeddings (nomic-embed-text)
│   │   └── vector_store.py         # FAISS vector store creation and management
│   │
│   ├── tools/                       # AI agent tools (functions agents can call)
│   │   ├── __init__.py             # Exports: search_f150_manual, search_web, set_vector_store
│   │   ├── manual_search.py        # RAG tool for searching F150 owner's manual
│   │   ├── web_search.py           # Brave Search API integration
│   │   ├── weather.py              # Weather tool (example)
│   │   └── location.py             # Location tool (example)
│   │
│   ├── utils/                       # Utility modules
│   │   ├── token_counter.py        # Ollama token tracking and context monitoring
│   │   └── conversation_memory.py  # Conversation memory manager for LangGraph
│   │
│   ├── config.py                    # Application configuration (Ollama, API keys, paths)
│   └── pdf/                         # PDF documents for RAG
│       └── 2018-Ford-F-150-Owners-Manual-version-5_om_EN-US_09_2018.pdf
│
├── tests/                           # Test files
├── wiki/                            # Documentation and planning
└── .env                             # Environment variables (not in git)
```

## Key Components

### Agent Code

#### LangChain Agent (`src/agent/f150_agent.py`)
- Uses `langchain.agents.create_agent()` high-level API
- Built on top of LangGraph (abstraction)
- Simple setup (~90 lines of code)
- System prompt defines agent behavior
- InMemorySaver checkpointer (conversation memory within session only)

#### LangGraph Agent (`src/graph/f150_graph.py`)
- Uses `langgraph.graph.StateGraph` for custom orchestration
- Explicit node and edge definitions
- Full control over agent loop (ReAct pattern)
- SqliteSaver checkpointer (persistent conversation memory)
- **Nodes**:
  - `agent`: Calls LLM to decide next action
  - `tools`: Executes tool calls (using ToolNode)
- **Edges**:
  - `START → agent`: Entry point
  - `agent → tools`: If LLM makes tool calls
  - `agent → END`: If LLM has final answer
  - `tools → agent`: After tools execute, loop back

### RAG (Retrieval-Augmented Generation)

Located in `src/rag/`:
- **Document Loader** (`document_loader.py`): Loads PDF, splits into chunks
- **Embeddings** (`embeddings.py`): Generates embeddings using Ollama (nomic-embed-text)
- **Vector Store** (`vector_store.py`): FAISS in-memory vector database for semantic search

**How it works**:
1. PDF chunked into 1000-character segments (200-char overlap)
2. Each chunk embedded using Ollama's nomic-embed-text model
3. Embeddings stored in FAISS for fast similarity search
4. Agent queries vector store to find relevant manual sections

### Tools (AI Agent Functions)

Located in `src/tools/`:

1. **`search_f150_manual`** (`manual_search.py`)
   - Searches 2018 F150 Owner's Manual using RAG
   - Returns top 5 most relevant manual sections
   - Includes page numbers and section metadata

2. **`search_web`** (`web_search.py`)
   - Searches the web using Brave Search API
   - Returns top 5 web results
   - Used for recalls, real-world fixes, current information

3. **Weather/Location tools** (examples, not used in F150 agent)

### Utilities

Located in `src/utils/`:

1. **`token_counter.py`** - Token Tracking
   - Monitors token usage via Ollama metadata
   - Tracks cumulative tokens across conversation
   - Warns when approaching 128k context limit
   - Works with both LangChain and LangGraph agents

2. **`conversation_memory.py`** - Memory Management (LangGraph only)
   - `ConversationMemoryManager` class
   - Generates unique thread IDs for sessions
   - Provides SqliteSaver checkpointer
   - Session listing and cleanup utilities
   - Only used by `main_graph.py` (LangGraph version)

### Configuration

**`src/config.py`** - Central configuration:
- **Ollama Settings**: Host, port, model selection
- **API Keys**: Brave Search API key
- **RAG Settings**: Chunk size, overlap, embedding model
- **LLM Settings**: Model (llama3.2), temperature (0 for deterministic)
- **Token Tracking**: Context limit (128k), warning threshold (80%)
- **Memory Settings**: SQLite database path for conversation persistence

**`.env`** file (create if missing):
```env
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3.2
BRAVE_API_KEY=your_brave_api_key_here
```

## LangChain vs LangGraph Comparison

| Aspect | LangChain (`main.py`) | LangGraph (`main_graph.py`) |
|--------|----------------------|----------------------------|
| **API Level** | High-level (`create_agent`) | Low-level (`StateGraph`) |
| **Lines of Code** | ~90 lines | ~180 lines |
| **Control** | Limited, abstracted | Full control over orchestration |
| **Memory Type** | In-memory (session only) | SQLite (persistent) |
| **Checkpointer** | `InMemorySaver` | `SqliteSaver` |
| **Agent Loop** | Hidden/automatic | Explicit nodes and edges |
| **Customization** | Via system prompt & middleware | Custom nodes, routing, state |
| **Best For** | Quick prototypes, simple agents | Production, complex workflows |
| **Memory Persistence** | ❌ Clears on restart | ✅ Survives restarts |

## Common Development Tasks

### Adding a New Tool

1. Create tool function in `src/tools/new_tool.py`:
```python
from langchain.tools import tool

@tool
def my_new_tool(query: str) -> str:
    """Tool description for the LLM."""
    # Implementation
    return result
```

2. Export in `src/tools/__init__.py`
3. Add to tools list in agent file (`f150_agent.py` or `f150_graph.py`)

### Modifying Agent Behavior

**LangChain version**: Edit system prompt in `src/agent/f150_agent.py`

**LangGraph version**: Edit nodes/edges/routing logic in `src/graph/f150_graph.py`

### Conversation Memory Management

**LangGraph only** (via `conversation_memory.py`):
```python
from src.utils.conversation_memory import ConversationMemoryManager

manager = ConversationMemoryManager("conversations.db")
conversations = manager.list_conversations()  # List all sessions
manager.delete_conversation(thread_id)  # Delete specific session
manager.clear_all_conversations()  # Clear all history
```

## Architecture Notes

### How Conversation Memory Works (LangGraph)

1. **Session Start**: Generate unique `thread_id` (UUID)
2. **Config**: Pass `{"configurable": {"thread_id": thread_id}}` to `agent.invoke()`
3. **Checkpointer**: SqliteSaver automatically saves state after each node execution
4. **State Persistence**: Full `MessagesState` (all messages) saved to SQLite
5. **Retrieval**: On next invoke with same `thread_id`, history automatically loaded
6. **Result**: Agent "remembers" entire conversation context

### Token Tracking

Both agents track tokens using Ollama's response metadata:
- **Prompt tokens**: Input to LLM (including conversation history)
- **Completion tokens**: LLM output
- **Total tokens**: Sum tracked against 128k context limit
- **Warning**: Displayed when context usage exceeds 80%

## Dependencies

Key packages (see requirements.txt):
- `langchain` - High-level agent framework
- `langgraph` - Low-level graph orchestration
- `langchain-ollama` - Ollama integration
- `langchain-community` - Community tools (Brave Search, FAISS)
- `faiss-cpu` - Vector similarity search
- `pypdf` - PDF document loading

## Development Workflow

1. **Ollama Setup**: Ollama runs on a separate machine on the local area network (not localhost)
   - Configure `OLLAMA_HOST` and `OLLAMA_PORT` in `.env` to point to the network machine
   - Ensure the following models are pulled on the Ollama host machine:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
   - Network connectivity to the Ollama host is required for all agent operations

2. **Set Environment**: Configure `.env` with API keys and Ollama settings

3. **Choose Framework**:
   - Prototyping? → Use LangChain (`main.py`)
   - Production/Custom? → Use LangGraph (`main_graph.py`)

4. **Test Changes**: Run appropriate entry point

5. **Memory Management**: LangGraph stores conversations in `conversations.db`

## Important Notes

- **Ollama Required**: Both agents require Ollama running locally or on network
- **Manual Required**: F150 PDF must exist at `src/pdf/2018-Ford-F-150-Owners-Manual...pdf`
- **Embeddings**: First run generates embeddings (~15-30 seconds)
- **Memory Trade-offs**: LangGraph persistent memory vs LangChain simplicity
- **Token Limits**: llama3.2 has 128k context, tracked automatically
- **Brave API**: Optional but recommended for web search capability
