"""
Unit tests for the ChromaDB vector store module.
"""

import shutil
from pathlib import Path
from src.chunking import chunk_pages
from src.vector_store import VectorStoreManager

TEST_DB_DIR = Path("data") / "test_chroma_db"


def cleanup():
    if TEST_DB_DIR.exists():
        shutil.rmtree(TEST_DB_DIR, ignore_errors=True)


def test_add_and_query():
    store = VectorStoreManager(persist_dir=TEST_DB_DIR, collection_name="test_coll_add")
    store.reset()

    pages = [
        {
            "page_number": 1,
            "text": "Polyester warp yarn tension is maintained at 12 cN to prevent shedding in rapier looms.",
            "source": "weaving_manual.pdf"
        },
        {
            "page_number": 2,
            "text": "Reactive dye fixing on cellulosic fiber requires alkaline bath at pH 10.5.",
            "source": "dyeing_handbook.pdf"
        }
    ]

    chunks = chunk_pages(pages, chunk_size=300, chunk_overlap=50)
    count = store.add_chunks(chunks)
    assert count == 2
    assert store.count() == 2

    # Query weaving
    results = store.query("What is the tension for polyester warp yarn on rapier looms?", top_k=1)
    assert len(results) == 1
    assert "rapier looms" in results[0]["text"]
    assert results[0]["metadata"]["source"] == "weaving_manual.pdf"
    assert results[0]["metadata"]["page_number"] == 1
    assert results[0]["similarity"] > 0.4
    print("PASS: test_add_and_query")


def test_duplicate_ingestion_idempotence():
    store = VectorStoreManager(persist_dir=TEST_DB_DIR, collection_name="test_coll_dup")
    store.reset()

    pages = [
        {"page_number": 1, "text": "Mercerization improves luster and tensile strength.", "source": "finishing.pdf"}
    ]
    chunks = chunk_pages(pages)

    # Ingest once
    store.add_chunks(chunks)
    assert store.count() == 1

    # Ingest exact same chunks again (should update, not duplicate)
    store.add_chunks(chunks)
    assert store.count() == 1
    print("PASS: test_duplicate_ingestion_idempotence")


def test_document_deletion():
    store = VectorStoreManager(persist_dir=TEST_DB_DIR, collection_name="test_coll_del")
    store.reset()

    pages = [
        {"page_number": 1, "text": "Wool scouring removes lanolin wax.", "source": "wool.pdf"},
        {"page_number": 2, "text": "Silk degumming removes sericin protein.", "source": "silk.pdf"}
    ]
    chunks = chunk_pages(pages)
    store.add_chunks(chunks)
    assert store.count() == 2

    sources = store.get_indexed_sources()
    assert "wool.pdf" in sources
    assert "silk.pdf" in sources

    # Delete wool.pdf
    remaining = store.delete_document("wool.pdf")
    assert remaining == 1
    assert store.get_indexed_sources() == ["silk.pdf"]

    # Verify wool is no longer retrievable
    res = store.query("lanolin wax", top_k=2)
    for match in res:
        assert match["metadata"]["source"] != "wool.pdf"

    print("PASS: test_document_deletion")


def main():
    print("Testing vector store module...\n")
    try:
        test_add_and_query()
        test_duplicate_ingestion_idempotence()
        test_document_deletion()
        print("\nAll vector store tests passed!")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
