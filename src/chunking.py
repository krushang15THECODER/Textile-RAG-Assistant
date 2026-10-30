from typing import Any, Dict, List


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """Split raw text into overlapping passages preserving natural word boundaries."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size.")

    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    chunks: List[str] = []
    start = 0
    text_len = len(cleaned_text)
    step = chunk_size - chunk_overlap

    while start < text_len:
        end = start + chunk_size

        if end >= text_len:
            final_chunk = cleaned_text[start:].strip()
            if final_chunk:
                chunks.append(final_chunk)
            break

        boundary = -1
        for separator in ["\n\n", "\n", " "]:
            found = cleaned_text.rfind(separator, start, end)
            if found > start + chunk_overlap:
                boundary = found + len(separator)
                break

        if boundary <= start:
            boundary = end

        chunk = cleaned_text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)

        next_start = boundary - chunk_overlap
        if next_start <= start:
            next_start = start + step
        start = next_start

    return chunks


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    default_source: str = "document.pdf"
) -> List[Dict[str, Any]]:
    """Chunk extracted pages with deterministic chunk IDs and provenance metadata."""
    chunks: List[Dict[str, Any]] = []

    for page in pages:
        text = page.get("text", "")
        page_number = page.get("page_number", 1)
        source = page.get("source", default_source)

        text_chunks = split_text(text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for chunk_index, chunk_text in enumerate(text_chunks):
            chunk_id = f"{source}_p{page_number}_c{chunk_index}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "char_count": len(chunk_text)
                }
            })

    return chunks
