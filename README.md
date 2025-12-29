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
   - OpenWeatherMap API key: https://openweathermap.org/api (free tier available)
   - Anthropic API key: https://console.anthropic.com/

3. Create a `.env` file in the project root:
```bash
cp .env.example .env
```

4. Add your API keys to the `.env` file:
```
OPENWEATHER_API_KEY=your_openweather_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## Usage

Run the weather agent:
```bash
uv run python main.py
```

Example interactions:
- "What's the weather like?" - Gets weather for Grand Rapids, Michigan
- "What's the weather in London?" - Gets weather for London
- "How's the weather in Tokyo, Japan?" - Gets weather for Tokyo
- Type `quit` or `exit` to end the session

## How It Works

The agent uses two tools:
1. `get_current_location()` - Returns the default location (Grand Rapids, Michigan)
2. `get_weather(location)` - Fetches current weather data from OpenWeatherMap API

The LangChain agent intelligently decides when to use each tool based on the user's question, retrieves the weather data, and presents it in a conversational format.

## Project Structure

```
langchain-tutorial/
├── main.py                          # Entry point and interactive CLI
├── src/
│   ├── config.py                    # Configuration and environment variables
│   ├── agent/
│   │   └── weather_agent.py         # Agent creation and setup
│   └── tools/
│       ├── weather.py               # Weather API tool
│       └── location.py              # Location tool
├── pyproject.toml                   # Project dependencies
└── .env.example                     # Template for environment variables
```

### Key Components

- [main.py](main.py) - Interactive CLI interface
- [src/config.py](src/config.py) - Centralized configuration management
- [src/agent/weather_agent.py](src/agent/weather_agent.py) - LangGraph agent setup using modern patterns
- [src/tools/weather.py](src/tools/weather.py) - Weather API integration tool
- [src/tools/location.py](src/tools/location.py) - Location retrieval tool
