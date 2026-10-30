from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingManager:
    """Manages model loading and vector encoding for documents and queries."""

    _instance: Optional["EmbeddingManager"] = None
    _model: Optional[SentenceTransformer] = None

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        if EmbeddingManager._model is None:
            try:
                EmbeddingManager._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                EmbeddingManager._model = SentenceTransformer(self.model_name)
        self.model = EmbeddingManager._model

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Compute normalized embeddings for a list of text strings."""
        if not texts:
            return []

        cleaned_texts = [t if t and t.strip() else " " for t in texts]
        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query into a 384-dimensional normalized vector."""
        if not query or not query.strip():
            return [0.0] * EMBEDDING_DIM

        embedding = self.model.encode(
            query.strip(),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.tolist()


_default_manager: Optional[EmbeddingManager] = None


def get_embedding_manager(model_name: str = DEFAULT_EMBEDDING_MODEL) -> EmbeddingManager:
    global _default_manager
    if _default_manager is None or _default_manager.model_name != model_name:
        _default_manager = EmbeddingManager(model_name=model_name)
    return _default_manager


def embed_texts(texts: List[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> List[List[float]]:
    return get_embedding_manager(model_name).embed_texts(texts)


def embed_query(query: str, model_name: str = DEFAULT_EMBEDDING_MODEL) -> List[float]:
    return get_embedding_manager(model_name).embed_query(query)
