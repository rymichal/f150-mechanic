import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration settings."""

    # Ollama settings
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
    OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

    # API Keys
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

    # LangSmith tracing settings
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "f150-expert-agent")

    # Default settings
    DEFAULT_LOCATION = "Grand Rapids, Michigan"

    # PDF/RAG settings
    PDF_DIRECTORY = os.path.join(os.path.dirname(__file__), "pdf")
    F150_MANUAL_PATH = os.path.join(PDF_DIRECTORY, "2018-Ford-F-150-Owners-Manual-version-5_om_EN-US_09_2018.pdf")

    # Chunking settings
    CHUNK_SIZE = 1000  # Characters per chunk
    CHUNK_OVERLAP = 200  # Overlap between chunks to preserve context

    # Embedding settings
    EMBEDDING_MODEL = "nomic-embed-text"  # Ollama embedding model
    # Other good options: mxbai-embed-large, all-minilm

    # LLM settings
    LLM_MODEL = OLLAMA_MODEL  # Use the model from environment
    LLM_TEMPERATURE = 0

    # Token tracking settings
    TOKEN_TRACKING_ENABLED = True
    CONTEXT_LIMIT = 128000  # Llama 3.2 has 128k context window
    TOKEN_WARNING_THRESHOLD = 80.0  # Warn when reaching 80% of context

    # Human-in-the-loop settings
    TOOL_APPROVAL_ENABLED = os.getenv("TOOL_APPROVAL_ENABLED", "false").lower() == "true"

    # Telemetry/Logging settings
    TELEMETRY = os.getenv("TELEMETRY", "true").lower() == "true"

    @classmethod
    def get_ollama_base_url(cls) -> str:
        """Construct the Ollama base URL from host and port."""
        return f"http://{cls.OLLAMA_HOST}:{cls.OLLAMA_PORT}"

    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.OLLAMA_HOST:
            print("Error: OLLAMA_HOST not found. Please set it in your .env file.")
            return False
        if not cls.OLLAMA_PORT:
            print("Error: OLLAMA_PORT not found. Please set it in your .env file.")
            return False
        return True
