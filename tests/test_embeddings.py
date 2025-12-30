"""Test script for embeddings."""

import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from src.rag.embeddings import test_embeddings


def main():
    print("=" * 70)
    print("STEP 2: Testing Ollama Embeddings")
    print("=" * 70)
    print()

    try:
        test_embeddings()
        print("\n" + "=" * 70)
        print("SUCCESS: Embeddings are working!")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running on your network")
        print("  2. You have pulled the embedding model:")
        print(f"     ollama pull nomic-embed-text")


if __name__ == "__main__":
    main()
