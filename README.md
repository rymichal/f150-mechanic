# LangChain F150 Agent Tutorial

A basic AI agent built with LangChain and Ollama that can advise on using a ford f150 truck.

## Features

- RAG with f150 user manual.
- Online searching
- Connects to network Ollama instance for on-prem AI.
- Defaults to use llama3.2 for natural language understanding

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Get API keys:
   - Brave Search API

3. Create a `.env` file in the project root:
```bash
cp .env.example .env
```

4. Add your API keys to the `.env` file:
```
OLLAMA_HOST=
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3.2
BRAVE_API_KEY=
```

## Usage

Run the F150 mechanic agent:
```bash
uv run python main.py
```

