from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings

from src.embeddings import embed_query, embed_texts

DEFAULT_PERSIST_DIR = Path("data") / "chroma_db"
DEFAULT_COLLECTION_NAME = "textile_knowledge"


class VectorStoreManager:
    """Manages persistent ChromaDB vector storage and semantic querying."""

    def __init__(
        self,
        persist_dir: Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Embed and upsert document chunks into the collection."""
        if not chunks:
            return 0

        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c.get("metadata", {}) for c in chunks]
        embeddings = embed_texts(texts)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        return len(ids)

    def query(
        self,
        query_text: str,
        top_k: int = 4,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search against indexed chunks."""
        if self.count() == 0 or not query_text or not query_text.strip():
            return []

        query_vec = embed_query(query_text)
        where_clause = {"source": source_filter} if source_filter else None

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_k, self.count()),
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
                similarity = max(0.0, 1.0 - dist)
                retrieved.append({
                    "chunk_id": chunk_id,
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                    "similarity": round(similarity, 4)
                })

        return retrieved

    def delete_document(self, source: str, document_id: Optional[str] = None) -> int:
        """Remove chunks matching source or document_id."""
        if document_id:
            try:
                self.collection.delete(where={"document_id": document_id})
            except Exception:
                pass
        if source:
            try:
                self.collection.delete(where={"source": source})
            except Exception:
                pass
        return self.count()

    def delete_by_document_id(self, document_id: str) -> int:
        """Remove chunks matching a specific document_id."""
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception:
            pass
        return self.count()

    def get_document_chunks_count(self, document_id: Optional[str] = None, source: Optional[str] = None) -> int:
        """Count chunks matching document_id or source."""
        try:
            where_clause = None
            if document_id:
                where_clause = {"document_id": document_id}
            elif source:
                where_clause = {"source": source}
            if where_clause:
                res = self.collection.get(where=where_clause, include=[])
                return len(res.get("ids", []))
        except Exception:
            pass
        return 0

    def count(self) -> int:
        """Return total chunks in collection."""
        return self.collection.count()

    def get_indexed_sources(self) -> List[str]:
        """Return sorted list of unique source filenames in collection."""
        if self.count() == 0:
            return []

        data = self.collection.get(include=["metadatas"])
        metas = data.get("metadatas", [])
        sources = {m.get("source") for m in metas if m and "source" in m}
        return sorted([s for s in sources if s])

    def reset(self):
        """Reset the vector collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )


_default_vector_store: Optional[VectorStoreManager] = None


def get_vector_store(
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME
) -> VectorStoreManager:
    global _default_vector_store
    if _default_vector_store is None:
        _default_vector_store = VectorStoreManager(
            persist_dir=persist_dir,
            collection_name=collection_name
        )
    return _default_vector_store
