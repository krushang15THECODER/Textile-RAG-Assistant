"""
Unit tests for the chunking module in Textile RAG Assistant.
"""

from src.chunking import chunk_pages, split_text


def test_empty_and_whitespace():
    """Verify that empty strings or whitespace-only inputs return no chunks."""
    assert split_text("") == []
    assert split_text("   \n\t  ") == []

    pages = [
        {"page_number": 1, "text": "   ", "source": "test.pdf"},
        {"page_number": 2, "text": "", "source": "test.pdf"},
    ]
    assert chunk_pages(pages) == []
    print("PASS: test_empty_and_whitespace")


def test_short_text():
    """Verify text shorter than chunk_size remains a single chunk."""
    text = "Polyester fiber exhibits high tensile strength and chemical resistance."
    chunks = split_text(text, chunk_size=500, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == text
    print("PASS: test_short_text")


def test_oversized_text_and_overlap():
    """Verify text splitting and overlap between adjacent chunks."""
    # Create repeated textile terminology sentences
    sentence = "Ring spinning produces higher strength yarn compared to rotor spinning. "
    text = sentence * 20  # ~1440 characters
    chunk_size = 300
    chunk_overlap = 60

    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(chunks) > 1

    for chunk in chunks:
        # Every chunk should respect the upper boundary (with slight allowance for boundary word cut)
        assert len(chunk) <= chunk_size + 10

    # Test that adjacent chunks have overlapping content
    for i in range(len(chunks) - 1):
        first_tail = chunks[i][-40:]
        second_head = chunks[i + 1][:80]
        # At least some words in common between the tail of chunk i and head of chunk i+1
        common_words = set(first_tail.split()) & set(second_head.split())
        assert len(common_words) > 0, f"No overlap found between chunk {i} and {i+1}"

    print(f"PASS: test_oversized_text_and_overlap ({len(chunks)} chunks generated)")


def test_unbroken_text_fallback():
    """Verify handling of continuous strings with no whitespace."""
    unbroken = "X" * 1200
    chunks = split_text(unbroken, chunk_size=400, chunk_overlap=100)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk) <= 400
    print("PASS: test_unbroken_text_fallback")


def test_metadata_preservation():
    """Verify source, page_number, chunk_index, and IDs are preserved."""
    pages = [
        {
            "page_number": 1,
            "text": "Carding is the process of untangling and intermixing fibers.",
            "source": "textile_processing_guide.pdf"
        },
        {
            "page_number": 2,
            "text": "Combing removes short fibers and impurities to create premium combed cotton yarn. " * 10,
            "source": "textile_processing_guide.pdf"
        }
    ]

    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=50)

    assert len(chunks) >= 2

    # Check page 1 chunk
    c0 = chunks[0]
    assert c0["chunk_id"] == "textile_processing_guide.pdf_p1_c0"
    assert c0["metadata"]["source"] == "textile_processing_guide.pdf"
    assert c0["metadata"]["page_number"] == 1
    assert c0["metadata"]["chunk_index"] == 0
    assert "Carding" in c0["text"]

    # Check page 2 chunks have page_number 2 and proper indexing
    p2_chunks = [c for c in chunks if c["metadata"]["page_number"] == 2]
    assert len(p2_chunks) > 1
    for idx, c in enumerate(p2_chunks):
        assert c["chunk_id"] == f"textile_processing_guide.pdf_p2_c{idx}"
        assert c["metadata"]["page_number"] == 2
        assert c["metadata"]["chunk_index"] == idx

    print("PASS: test_metadata_preservation")


def main():
    print("Running Milestone 1 Chunking tests...\n")
    test_empty_and_whitespace()
    test_short_text()
    test_oversized_text_and_overlap()
    test_unbroken_text_fallback()
    test_metadata_preservation()
    print("\nAll chunking tests passed successfully!")


if __name__ == "__main__":
    main()
