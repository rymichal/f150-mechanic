"""
Embedding utilities for converting text to vectors.

Embeddings capture semantic meaning - similar text gets similar vectors.
This enables semantic search: finding relevant chunks based on meaning, not just keywords.
"""

from langchain_ollama import OllamaEmbeddings
from ..config import Config


def create_embeddings():
    """
    Create an embeddings instance using Ollama.

    Returns:
        OllamaEmbeddings instance configured with your Ollama server.

    Example:
        >>> embeddings = create_embeddings()
        >>> vector = embeddings.embed_query("What is fuse 33?")
        >>> print(len(vector))  # Should be 768 for nomic-embed-text
        768
    """
    embeddings = OllamaEmbeddings(
        model=Config.EMBEDDING_MODEL,
        base_url=Config.get_ollama_base_url()
    )

    return embeddings


def test_embeddings():
    """
    Test the embedding model to verify it's working.

    This will:
    1. Connect to Ollama
    2. Generate embeddings for sample text
    3. Show embedding dimensions and similarity
    """
    print("Testing Ollama embeddings...\n")
    print(f"Model: {Config.EMBEDDING_MODEL}")
    print(f"Ollama URL: {Config.get_ollama_base_url()}\n")

    embeddings = create_embeddings()

    # Test with sample queries
    test_queries = [
        "What is fuse 33 for?",
        "Where is fuse number 33?",
        "How do I check tire pressure?",
    ]

    print("Generating embeddings for test queries...")

    vectors = []
    for query in test_queries:
        print(f"  - '{query}'")
        vector = embeddings.embed_query(query)
        vectors.append(vector)

    # Show embedding info
    print(f"\n✓ Successfully generated {len(vectors)} embeddings")
    print(f"✓ Vector dimension: {len(vectors[0])}")

    # Calculate similarity between first two queries (about fuses)
    # vs similarity with third query (about tires)
    from numpy import dot
    from numpy.linalg import norm

    def cosine_similarity(a, b):
        """Calculate cosine similarity between two vectors (0-1, higher = more similar)."""
        return dot(a, b) / (norm(a) * norm(b))

    sim_fuse = cosine_similarity(vectors[0], vectors[1])
    sim_different = cosine_similarity(vectors[0], vectors[2])

    print(f"\n✓ Similarity between fuse queries: {sim_fuse:.3f}")
    print(f"✓ Similarity fuse vs tire query: {sim_different:.3f}")

    if sim_fuse > sim_different:
        print("\n✓ Embeddings working correctly! Similar queries have higher similarity.")
    else:
        print("\n⚠ Warning: Similarity scores unexpected.")

    return embeddings


if __name__ == "__main__":
    """Test embeddings."""
    test_embeddings()
