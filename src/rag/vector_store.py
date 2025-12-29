"""
Vector store creation and management using FAISS.

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search
and clustering of dense vectors. Perfect for RAG!

Why FAISS:
- Fast similarity search (millions of vectors)
- In-memory or persistent
- No dependencies on external services
- Works great with Python 3.14
"""

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .document_loader import load_and_chunk_pdf
from .embeddings import create_embeddings
from ..config import Config


def create_vector_store(chunks: List[Document] = None):
    """
    Create a FAISS vector store from document chunks.

    Args:
        chunks: List of Document objects. If None, loads F150 manual automatically.

    Returns:
        FAISS vector store with indexed chunks

    Example:
        >>> vector_store = create_vector_store()
        >>> results = vector_store.similarity_search("What is fuse 33?", k=5)
        >>> print(results[0].page_content)
    """
    # Load chunks if not provided
    if chunks is None:
        print("Loading and chunking PDF...")
        chunks = load_and_chunk_pdf()

    print(f"Creating vector store from {len(chunks)} chunks...")

    # Create embeddings instance
    embeddings = create_embeddings()

    # Create FAISS vector store
    # This will:
    # 1. Embed all chunks (convert text → vectors)
    # 2. Build FAISS index for fast similarity search
    # 3. Store both vectors and original text
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    print(f"✓ Vector store created successfully!")
    print(f"  - {len(chunks)} chunks indexed")
    print(f"  - Ready for similarity search")

    return vector_store


def save_vector_store(vector_store, path: str = "f150_vector_store"):
    """
    Save vector store to disk for later use.

    Args:
        vector_store: FAISS vector store to save
        path: Directory path to save to (default: f150_vector_store)
    """
    vector_store.save_local(path)
    print(f"✓ Vector store saved to {path}")


def load_vector_store(path: str = "f150_vector_store"):
    """
    Load a previously saved vector store from disk.

    Args:
        path: Directory path to load from

    Returns:
        Loaded FAISS vector store
    """
    embeddings = create_embeddings()
    vector_store = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True  # We trust our own data
    )
    print(f"✓ Vector store loaded from {path}")
    return vector_store


def test_vector_store(vector_store):
    """
    Test the vector store with sample queries.

    Args:
        vector_store: FAISS vector store to test
    """
    print("\n" + "=" * 70)
    print("TESTING VECTOR STORE - Similarity Search")
    print("=" * 70)

    test_queries = [
        "What is fuse 33 for?",
        "How do I check tire pressure?",
        "What is the towing capacity?",
    ]

    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 70)

        # Search for top 3 most similar chunks
        results = vector_store.similarity_search(query, k=3)

        print(f"Found {len(results)} relevant chunks:\n")

        for i, doc in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Page: {doc.metadata.get('page', 'unknown')}")
            print(f"  Content preview: {doc.page_content[:150]}...")
            print()


if __name__ == "__main__":
    """Test vector store creation and search."""
    print("=" * 70)
    print("STEP 3: Creating Vector Store with FAISS")
    print("=" * 70)
    print()

    # Create vector store
    vector_store = create_vector_store()

    # Test similarity search
    test_vector_store(vector_store)

    print("\n" + "=" * 70)
    print("SUCCESS: Vector store is working!")
    print("=" * 70)
