import html
from typing import Any, Dict, List

ABSTENTION_PHRASES = [
    "do not contain",
    "does not contain",
    "cannot answer",
    "not mentioned in the provided",
    "insufficient information",
    "insufficient evidence",
    "not found in the uploaded documents",
    "no information",
    "not provided in the context"
]


def extract_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract deduplicated citations from retrieved evidence chunks."""
    citations = []
    seen = set()

    for chunk in chunks:
        source = chunk.get("source", "unknown")
        page = chunk.get("page_number", 1)
        chunk_id = chunk.get("chunk_id", f"{source}_p{page}")
        sim = chunk.get("similarity", 0.0)
        text = chunk.get("text", "").strip()

        dedup_key = (source, page)
        if dedup_key not in seen:
            seen.add(dedup_key)
            citations.append({
                "source": source,
                "page_number": page,
                "chunk_id": chunk_id,
                "similarity": sim,
                "snippet": text[:180] + "..." if len(text) > 180 else text
            })

    return citations


def is_abstention(answer: str) -> bool:
    """Check if generated response is an abstention due to insufficient context."""
    lowered = answer.lower()
    return any(phrase in lowered for phrase in ABSTENTION_PHRASES)


def format_citations_markdown(citations: List[Dict[str, Any]]) -> str:
    """Format citation list into clean markdown for assistant UI display."""
    if not citations:
        return ""

    lines = []
    for idx, c in enumerate(citations, start=1):
        source = c["source"]
        page = c["page_number"]
        sim_pct = int(c.get("similarity", 0.0) * 100)
        snippet = c.get("snippet", "").replace("\n", " ").strip()

        lines.append(f"- **[{idx}] {source}** — *Page {page}* (Relevance: `{sim_pct}%`)")
        if snippet:
            lines.append(f"  > \"{snippet}\"\n")

    return "\n".join(lines)


# Alias to maintain full backwards compatibility without raw HTML rendering issues
format_citations_html = format_citations_markdown


