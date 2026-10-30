"""
Unit tests for embeddings module.
"""

from src.embeddings import (
    EmbeddingManager,
    embed_query,
    embed_texts,
    EMBEDDING_DIM
)


def test_embedding_dimensions():
    manager = EmbeddingManager()
    texts = [
        "Cotton yarn tensile strength specification.",
        "Ring spinning vs open-end rotor spinning efficiency."
    ]
    vectors = manager.embed_texts(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) == EMBEDDING_DIM
    assert len(vectors[1]) == EMBEDDING_DIM
    print(f"PASS: test_embedding_dimensions (dim={EMBEDDING_DIM})")


def test_empty_inputs():
    manager = EmbeddingManager()
    empty_batch = manager.embed_texts([])
    assert empty_batch == []

    empty_query = manager.embed_query("")
    assert len(empty_query) == EMBEDDING_DIM
    assert all(v == 0.0 for v in empty_query)
    print("PASS: test_empty_inputs")


def test_semantic_consistency():
    q = "What is cotton fiber strength?"
    similar_doc = "Tensile testing and breaking force of natural cotton fibers."
    different_doc = "Digital dye sublimation printer maintenance and ink cartridges."

    q_vec = embed_query(q)
    docs_vec = embed_texts([similar_doc, different_doc])

    # Dot product of normalized vectors = cosine similarity
    sim_similar = sum(a * b for a, b in zip(q_vec, docs_vec[0]))
    sim_different = sum(a * b for a, b in zip(q_vec, docs_vec[1]))

    assert sim_similar > sim_different
    print(f"PASS: test_semantic_consistency (similar={sim_similar:.3f} > diff={sim_different:.3f})")


def main():
    print("Testing embeddings module...\n")
    test_embedding_dimensions()
    test_empty_inputs()
    test_semantic_consistency()
    print("\nAll embedding tests passed!")


if __name__ == "__main__":
    main()
