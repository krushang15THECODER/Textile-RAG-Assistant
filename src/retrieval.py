from typing import Any, Dict, List, Optional
from src.vector_store import VectorStoreManager, get_vector_store


class Retriever:
    """Handles semantic retrieval of document chunks with similarity filtering."""

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or get_vector_store()

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: Optional[float] = None,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant text chunks for a user query."""
        if not query or not query.strip():
            return []

        raw_results = self.vector_store.query(
            query_text=query.strip(),
            top_k=top_k,
            source_filter=source_filter
        )

        formatted_results = []
        for item in raw_results:
            similarity = item.get("similarity", 0.0)
            if score_threshold is not None and similarity < score_threshold:
                continue

            metadata = item.get("metadata", {})
            formatted_results.append({
                "chunk_id": item.get("chunk_id"),
                "text": item.get("text", ""),
                "source": metadata.get("source", "unknown"),
                "page_number": metadata.get("page_number", 1),
                "chunk_index": metadata.get("chunk_index", 0),
                "similarity": similarity,
                "distance": item.get("distance", 1.0),
                "metadata": metadata
            })

        return formatted_results


def retrieve_context(
    query: str,
    top_k: int = 4,
    score_threshold: Optional[float] = None,
    vector_store: Optional[VectorStoreManager] = None
) -> List[Dict[str, Any]]:
    return Retriever(vector_store=vector_store).retrieve(
        query=query, top_k=top_k, score_threshold=score_threshold
    )


def format_context_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block for LLM prompting."""
    if not chunks:
        return "No relevant context found."

    context_blocks = []
    for idx, c in enumerate(chunks, start=1):
        source = c.get("source", "unknown")
        page = c.get("page_number", 1)
        text = c.get("text", "").strip()
        context_blocks.append(f"[Passage {idx} | Source: {source} | Page: {page}]\n{text}")

    return "\n\n".join(context_blocks)
