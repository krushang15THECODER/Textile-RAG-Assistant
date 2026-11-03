import gc
import os
from typing import Any, List, Optional
import numpy as np

# Strict thread limits to keep memory footprint under 256MB on cloud hosts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingManager:
    """Manages model loading and vector encoding for documents and queries."""

    _instance: Optional["EmbeddingManager"] = None
    _model: Optional[Any] = None

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        if EmbeddingManager._model is None:
            import torch
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass

            from sentence_transformers import SentenceTransformer
            try:
                EmbeddingManager._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                EmbeddingManager._model = SentenceTransformer(self.model_name)
        self.model = EmbeddingManager._model

    def embed_texts(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Compute normalized embeddings for a list of text strings in small batches to save RAM."""
        if not texts:
            return []

        import torch
        cleaned_texts = [t if t and t.strip() else " " for t in texts]
        with torch.inference_mode():
            embeddings = self.model.encode(
                cleaned_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
        gc.collect()
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query into a 384-dimensional normalized vector."""
        if not query or not query.strip():
            return [0.0] * EMBEDDING_DIM

        import torch
        with torch.inference_mode():
            embedding = self.model.encode(
                query.strip(),
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
        gc.collect()
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
