import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.models import DocumentItemResponse
from backend.security import require_admin
from src.chunking import chunk_pages
from src.ingestion import extract_text_from_bytes
from src.vector_store import get_vector_store

router = APIRouter(prefix="/api/admin", tags=["Admin Operations"])

DOCUMENTS_DIR = Path("data") / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = Path("data") / "documents_registry.json"


def load_documents_registry() -> Dict[str, Dict[str, Any]]:
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_documents_registry(registry: Dict[str, Dict[str, Any]]) -> None:
    try:
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        print(f"Error saving registry: {e}")


@router.get("/documents", response_model=List[DocumentItemResponse])
async def list_documents(admin_user: dict = Depends(require_admin)):
    """Retrieve all indexed technical documents and vector statistics (Admin only)."""
    registry = load_documents_registry()
    items = []
    for doc_id, doc in registry.items():
        items.append(
            DocumentItemResponse(
                document_id=doc_id,
                filename=doc.get("filename", "unknown"),
                file_size=doc.get("file_size", 0),
                upload_date=doc.get("upload_date", ""),
                chunk_count=doc.get("chunk_count", 0),
                page_count=doc.get("page_count", "Indexed"),
                status=doc.get("status", "Indexed")
            )
        )
    return items


@router.post("/documents/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    admin_user: dict = Depends(require_admin)
):
    """Upload a technical PDF, parse text, embed, and store in ChromaDB (Admin only)."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF technical documents are accepted."
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded PDF file is empty."
        )

    vs = get_vector_store()
    doc_id = f"doc_{hashlib.md5(file.filename.encode()).hexdigest()[:10]}"

    # Save to disk
    dest_path = DOCUMENTS_DIR / f"{doc_id}_{file.filename}"
    with open(dest_path, "wb") as f:
        f.write(file_bytes)

    # Extract text with PyMuPDF
    try:
        pages = extract_text_from_bytes(file_bytes, file.filename)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to extract text from PDF: {e}"
        )

    if not pages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract any readable text pages from the PDF."
        )

    # Chunk pages
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=100, default_source=file.filename)
    for c in chunks:
        c["metadata"]["document_id"] = doc_id
        c["metadata"]["source"] = file.filename

    # Delete existing if re-uploading
    try:
        vs.delete_document(file.filename)
    except Exception:
        pass

    added_chunks = vs.add_chunks(chunks)

    # Update registry
    registry = load_documents_registry()
    registry[doc_id] = {
        "document_id": doc_id,
        "filename": file.filename,
        "file_path": str(dest_path),
        "file_size": len(file_bytes),
        "upload_date": datetime.now().strftime("%Y-%m-%d"),
        "chunk_count": added_chunks,
        "page_count": len(pages),
        "status": "Indexed"
    }
    save_documents_registry(registry)

    return {
        "document_id": doc_id,
        "filename": file.filename,
        "chunk_count": added_chunks,
        "page_count": len(pages),
        "message": f"Successfully indexed {added_chunks} chunks from '{file.filename}'."
    }


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    admin_user: dict = Depends(require_admin)
):
    """Purge a document from the vector store and disk archive (Admin only)."""
    registry = load_documents_registry()
    if document_id not in registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found in registry."
        )

    doc_info = registry[document_id]
    filename = doc_info.get("filename", "")
    file_path = doc_info.get("file_path", "")

    vs = get_vector_store()
    try:
        if filename:
            vs.delete_document(filename)
    except Exception as e:
        print(f"Warning deleting from vector store: {e}")

    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass

    registry.pop(document_id, None)
    save_documents_registry(registry)

    return {
        "message": f"Document '{filename}' and its chunks have been successfully purged from the knowledge base."
    }
