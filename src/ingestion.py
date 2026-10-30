from pathlib import Path
import pymupdf


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from a PDF while preserving page numbers."""
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    pages = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text
                })

    return pages


def extract_text_from_bytes(
    pdf_bytes: bytes,
    filename: str = "uploaded.pdf"
) -> list[dict]:
    """Extract page-level text directly from uploaded PDF bytes."""

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    pages = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page_number": page_number,
                    "text": text
                })

    return pages
