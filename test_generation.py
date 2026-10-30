"""
Unit tests for generation and citations module.
"""

from src.citations import extract_citations, format_citations_markdown, is_abstention
from src.generation import generate_answer


def test_citations_extraction():
    chunks = [
        {
            "chunk_id": "spinning.pdf_p1_c0",
            "text": "Ring spinning imparts twist using a traveler revolving around a ring.",
            "source": "spinning.pdf",
            "page_number": 1,
            "similarity": 0.88
        },
        {
            "chunk_id": "spinning.pdf_p1_c1",
            "text": "Ring spinning travelers require lubrication at high spindle speeds.",
            "source": "spinning.pdf",
            "page_number": 1,
            "similarity": 0.82
        },
        {
            "chunk_id": "weaving.pdf_p4_c0",
            "text": "Warp beam preparation requires uniform tensioning.",
            "source": "weaving.pdf",
            "page_number": 4,
            "similarity": 0.75
        }
    ]

    cites = extract_citations(chunks)
    # spinning.pdf page 1 should be deduplicated
    assert len(cites) == 2
    assert cites[0]["source"] == "spinning.pdf"
    assert cites[0]["page_number"] == 1
    assert cites[1]["source"] == "weaving.pdf"
    assert cites[1]["page_number"] == 4

    md = format_citations_markdown(cites)
    assert "**[1] spinning.pdf**" in md
    assert "*Page 1*" in md
    print("PASS: test_citations_extraction")


def test_abstention_detection():
    assert is_abstention("The provided documents do not contain sufficient information to answer.")
    assert is_abstention("I cannot answer this based on the uploaded documents.")
    assert is_abstention("This is not mentioned in the provided context.")
    assert not is_abstention("Cotton fiber breaking tenacity is 28 cN/tex according to the report.")
    print("PASS: test_abstention_detection")


def test_empty_context_abstention():
    # Calling generate_answer with empty chunks must abstain immediately without invoking the API
    result = generate_answer("What is the warp yarn count?", retrieved_chunks=[])
    assert result["abstained"] is True
    assert result["is_grounded"] is False
    assert "do not contain sufficient information" in result["answer"]
    print("PASS: test_empty_context_abstention")


def test_missing_api_key_error_handling():
    # When api_key is empty and not in env, should return graceful error dict, not crash
    result = generate_answer(
        "What is ring spinning?",
        retrieved_chunks=[{"source": "test.pdf", "page_number": 1, "text": "Sample text", "similarity": 0.8}],
        api_key=""
    )
    assert result["error"] is not None
    assert "GEMINI_API_KEY is missing or invalid" in result["answer"]
    print("PASS: test_missing_api_key_error_handling")


def main():
    print("Testing citations and generation...\n")
    test_citations_extraction()
    test_abstention_detection()
    test_empty_context_abstention()
    test_missing_api_key_error_handling()
    print("\nAll generation & citation tests passed!")


if __name__ == "__main__":
    main()
