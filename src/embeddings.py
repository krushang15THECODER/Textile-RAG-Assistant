import gc
import os
from pathlib import Path
from typing import Any, List, Optional
import numpy as np

# Strict thread limits to keep memory footprint low on cloud hosts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class ONNXModelWrapper:
    """Ultra-lightweight ONNX Runtime embedding engine (<60MB RAM footprint)."""

    def __init__(self, model_path: str, tokenizer_path: str):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=256)

    def encode(self, texts, batch_size: int = 16, **kwargs):
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts
        if not text_list:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        all_vectors = []
        for i in range(0, len(text_list), batch_size):
            batch = [t if t and t.strip() else " " for t in text_list[i : i + batch_size]]
            encodings = self.tokenizer.encode_batch(batch)

            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

            outputs = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )
            token_embeddings = outputs[0]
            mask_expanded = np.broadcast_to(
                np.expand_dims(attention_mask, -1), token_embeddings.shape
            ).astype(float)
            sum_embeddings = np.sum(token_embeddings * mask_expanded, 1)
            sum_mask = np.clip(mask_expanded.sum(1), a_min=1e-9, a_max=None)
            pooled = sum_embeddings / sum_mask

            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            normalized = pooled / np.clip(norms, a_min=1e-12, a_max=None)
            all_vectors.extend(normalized)

        result = np.array(all_vectors, dtype=np.float32)
        return result[0] if is_single else result


class EmbeddingManager:
    """Manages model loading and vector encoding for documents and queries."""

    _instance: Optional["EmbeddingManager"] = None
    _model: Optional[Any] = None

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        if EmbeddingManager._model is not None:
            self.model = EmbeddingManager._model
            return

        # 1. Try local bundled ONNX model first (fastest, lowest memory, zero network)
        local_onnx = Path("models") / "all-MiniLM-L6-v2" / "model.onnx"
        local_tok = Path("models") / "all-MiniLM-L6-v2" / "tokenizer.json"
        if local_onnx.exists() and local_tok.exists():
            try:
                EmbeddingManager._model = ONNXModelWrapper(str(local_onnx), str(local_tok))
                self.model = EmbeddingManager._model
                return
            except Exception as onnx_err:
                print(f"Note: Local ONNX model failed: {onnx_err}")

        # 2. Try downloading ONNX from HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download
            hf_onnx = hf_hub_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", filename="onnx/model.onnx")
            hf_tok = hf_hub_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", filename="tokenizer.json")
            EmbeddingManager._model = ONNXModelWrapper(hf_onnx, hf_tok)
            self.model = EmbeddingManager._model
            return
        except Exception:
            pass

        # 3. Fallback to standard SentenceTransformer with PyTorch
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass

        from sentence_transformers import SentenceTransformer
        try:
            EmbeddingManager._model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            EmbeddingManager._model = SentenceTransformer(self.model_name)
        self.model = EmbeddingManager._model

    def embed_texts(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Compute normalized embeddings for a list of text strings."""
        if not texts:
            return []

        cleaned_texts = [t if t and t.strip() else " " for t in texts]
        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        gc.collect()
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(map(float, row)) for row in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """Embed a single search query into a 384-dimensional normalized vector."""
        if not query or not query.strip():
            return [0.0] * EMBEDDING_DIM

        embedding = self.model.encode(
            query.strip(),
            show_progress_bar=False,
            normalize_embeddings=True
        )
        gc.collect()
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(map(float, embedding))


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
