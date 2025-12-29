"""
Document loading and chunking for RAG (Retrieval-Augmented Generation).

This module handles:
1. Loading PDF documents
2. Splitting them into manageable chunks
3. Preserving metadata for citation/reference
"""

from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..config import Config


def load_and_chunk_pdf(pdf_path: str = None) -> List[Document]:
    """
    Load a PDF file and split it into chunks for RAG.

    Args:
        pdf_path: Path to PDF file. Defaults to F150 manual path from config.

    Returns:
        List of Document objects, each containing:
        - page_content: The text chunk
        - metadata: Dict with 'source' (file path) and 'page' (page number)

    Example:
        >>> chunks = load_and_chunk_pdf()
        >>> print(f"Loaded {len(chunks)} chunks")
        >>> print(chunks[0].page_content[:100])  # First 100 chars of first chunk
    """
    # Use default path if not provided
    if pdf_path is None:
        pdf_path = Config.F150_MANUAL_PATH

    print(f"Loading PDF from: {pdf_path}")

    # Load the PDF
    # PyPDFLoader extracts text page-by-page and preserves page numbers
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages from PDF")

    # Split documents into chunks
    # RecursiveCharacterTextSplitter tries to split on natural boundaries:
    # 1. Paragraphs (\n\n)
    # 2. Sentences (\n)
    # 3. Words (spaces)
    # 4. Characters (last resort)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,  # Use character count
        is_separator_regex=False,
    )

    # Split all documents into chunks
    chunks = text_splitter.split_documents(documents)

    print(f"Split into {len(chunks)} chunks")
    print(f"Chunk size: {Config.CHUNK_SIZE} chars, overlap: {Config.CHUNK_OVERLAP} chars")

    return chunks


def preview_chunks(chunks: List[Document], num_samples: int = 3):
    """
    Print preview of document chunks for inspection.

    Args:
        chunks: List of Document objects from load_and_chunk_pdf()
        num_samples: Number of random chunks to display
    """
    import random

    print("\n" + "=" * 70)
    print("CHUNK PREVIEW")
    print("=" * 70)

    sample_chunks = random.sample(chunks, min(num_samples, len(chunks)))

    for i, chunk in enumerate(sample_chunks, 1):
        print(f"\nSample {i}:")
        print(f"  Source: {chunk.metadata.get('source', 'unknown')}")
        print(f"  Page: {chunk.metadata.get('page', 'unknown')}")
        print(f"  Length: {len(chunk.page_content)} characters")
        print(f"  Content preview:")
        print(f"  {chunk.page_content[:200]}...")
        print("-" * 70)


if __name__ == "__main__":
    """Test the document loader."""
    print("Testing PDF loading and chunking...\n")

    # Load and chunk the F150 manual
    chunks = load_and_chunk_pdf()

    # Show statistics
    print(f"\nTotal chunks created: {len(chunks)}")

    if chunks:
        avg_length = sum(len(c.page_content) for c in chunks) / len(chunks)
        print(f"Average chunk length: {avg_length:.0f} characters")

        # Preview some chunks
        preview_chunks(chunks)
    else:
        print("No chunks were created. Check if the PDF file exists.")
