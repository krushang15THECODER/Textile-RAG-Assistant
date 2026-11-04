import os
from typing import Any, Dict, List, Optional
from src.vector_store import VectorStoreManager, get_vector_store

DEFAULT_SCORE_THRESHOLD = 0.40


def get_configured_threshold() -> float:
    """Read the retrieval score threshold from environment with fallback to 0.40."""
    raw = os.getenv("RETRIEVAL_SCORE_THRESHOLD", "").strip()
    if raw:
        try:
            return max(0.0, min(1.0, float(raw)))
        except ValueError:
            pass
    return DEFAULT_SCORE_THRESHOLD


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
        """Retrieve relevant text chunks for a user query.
        
        Args:
            query: The search query text.
            top_k: Maximum number of chunks to retrieve.
            score_threshold: Minimum cosine similarity (0.0 to 1.0). If None, defaults
                             to the configured system threshold (0.40).
            source_filter: Optional filename filter.
            
        Returns:
            List of formatted chunk dictionaries passing the similarity threshold.
        """
        if not query or not query.strip():
            return []

        active_threshold = score_threshold if score_threshold is not None else get_configured_threshold()

        raw_results = self.vector_store.query(
            query_text=query.strip(),
            top_k=top_k,
            source_filter=source_filter
        )

        formatted_results = []
        for item in raw_results:
            similarity = item.get("similarity", 0.0)
            if active_threshold > 0.0 and similarity < active_threshold:
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

        # Ensure sorted by similarity descending
        formatted_results.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        return formatted_results


def retrieve_context(
    query: str,
    top_k: int = 4,
    score_threshold: Optional[float] = None,
    vector_store: Optional[VectorStoreManager] = None,
    source_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    return Retriever(vector_store=vector_store).retrieve(
        query=query, top_k=top_k, score_threshold=score_threshold, source_filter=source_filter
    )


def format_context_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block for LLM prompting."""
    if not chunks:
        return "No relevant context found."

    context_blocks = []
    for idx, c in enumerate(chunks, start=1):
        source = c.get("source", "unknown")
        page = c.get("page_number", 1)
        similarity = c.get("similarity", 0.0)
        text = c.get("text", "").strip()
        context_blocks.append(
            f"[Passage {idx} | Source: {source} | Page: {page}]\n{text}"
        )

    return "\n\n".join(context_blocks)
