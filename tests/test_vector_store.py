"""Test script for vector store creation and search."""

import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from src.rag.vector_store import create_vector_store, test_vector_store


def main():
    print("=" * 70)
    print("STEP 3: Testing FAISS Vector Store")
    print("=" * 70)
    print()
    print("This will:")
    print("  1. Load the F150 manual PDF (641 pages)")
    print("  2. Split into chunks (1649 chunks)")
    print("  3. Generate embeddings for each chunk")
    print("  4. Build FAISS index for similarity search")
    print("  5. Test with sample queries")
    print()
    print("⏳ This may take 2-5 minutes (embedding 1649 chunks)...")
    print()

    try:
        # Create vector store (this does all the embedding)
        vector_store = create_vector_store()

        # Test similarity search
        test_vector_store(vector_store)

        print("\n" + "=" * 70)
        print("✓ SUCCESS: Vector store is working!")
        print("=" * 70)
        print("\nYou can now:")
        print("  - Search the manual semantically")
        print("  - Find relevant chunks for any question")
        print("  - Use this in the F150 agent")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running on your network")
        print("  2. You have pulled the embedding model:")
        print("     ollama pull nomic-embed-text")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
