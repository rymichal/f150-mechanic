# LangChain Weather Agent Tutorial

A basic AI agent built with LangChain and Claude that can check the weather for any location.

## Features

- Check current weather for any city
- Get weather for user's current location (defaults to Grand Rapids, Michigan)
- Interactive chat interface
- Uses Claude 3.5 Sonnet for natural language understanding
- Integrates with OpenWeatherMap API for real-time weather data

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

