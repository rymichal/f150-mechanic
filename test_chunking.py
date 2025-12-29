"""Test script for PDF loading and chunking."""

import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from src.rag.document_loader import load_and_chunk_pdf, preview_chunks


def main():
    print("Testing PDF loading and chunking...\n")

    # Load and chunk the F150 manual
    chunks = load_and_chunk_pdf()

    # Show statistics
    print(f"\n✓ Total chunks created: {len(chunks)}")

    if chunks:
        avg_length = sum(len(c.page_content) for c in chunks) / len(chunks)
        print(f"✓ Average chunk length: {avg_length:.0f} characters")

        # Preview some chunks
        preview_chunks(chunks, num_samples=3)

        print("\n" + "=" * 70)
        print("SUCCESS: PDF loading and chunking works!")
        print("=" * 70)
    else:
        print("ERROR: No chunks were created. Check if the PDF file exists.")


if __name__ == "__main__":
    main()
