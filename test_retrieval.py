"""
Unit tests for retrieval module.
"""

from pathlib import Path
from src.chunking import chunk_pages
from src.retrieval import Retriever, format_context_for_prompt
from src.vector_store import VectorStoreManager

TEST_DB_DIR = Path("data") / "test_retrieval_db"


def test_retrieval_flow():
    store = VectorStoreManager(persist_dir=TEST_DB_DIR, collection_name="test_retrieval")
    store.reset()

    # Empty store test
    retriever = Retriever(vector_store=store)
    empty_res = retriever.retrieve("What is cotton fiber length?")
    assert empty_res == []

    # Ingest mock textile data
    pages = [
        {
            "page_number": 1,
            "text": "Staple length of Egyptian Giza cotton averages 34mm to 38mm, conferring superior fineness.",
            "source": "cotton_grades.pdf"
        },
        {
            "page_number": 2,
            "text": "Short staple upland cotton measures under 26mm and is commonly used in coarse denim weaves.",
            "source": "cotton_grades.pdf"
        },
        {
            "page_number": 1,
            "text": "Disperse dyes are non-ionic dyes with low water solubility used for hydrophobic synthetic fibers.",
            "source": "dyeing_chemistry.pdf"
        }
    ]
    chunks = chunk_pages(pages, chunk_size=300, chunk_overlap=50)
    store.add_chunks(chunks)

    # Test top-k retrieval for staple length
    results = retriever.retrieve("What is the staple length of Egyptian Giza cotton?", top_k=2)
    assert len(results) >= 1
    top_match = results[0]
    assert "Giza cotton" in top_match["text"]
    assert top_match["source"] == "cotton_grades.pdf"
    assert top_match["page_number"] == 1
    assert "similarity" in top_match

    # Test format_context_for_prompt
    formatted = format_context_for_prompt(results)
    assert "[Passage 1 | Source: cotton_grades.pdf | Page: 1]" in formatted
    assert "Egyptian Giza cotton" in formatted

    # Test score threshold filtering
    high_threshold_results = retriever.retrieve("Unrelated topic about aerospace titanium alloy", score_threshold=0.8)
    assert len(high_threshold_results) == 0

    print("PASS: test_retrieval_flow")


def main():
    print("Testing retrieval module...\n")
    test_retrieval_flow()
    print("\nAll retrieval tests passed!")


if __name__ == "__main__":
    main()
