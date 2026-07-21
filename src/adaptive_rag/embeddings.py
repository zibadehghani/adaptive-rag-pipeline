from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from adaptive_rag.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_LENGTH,
    HUGGINGFACE_CACHE_DIR,
    QWEN_MODEL_ID,
)


class Qwen3Embeddings(Embeddings):
    """LangChain-compatible wrapper for Qwen3 Embedding."""

    def __init__(
        self,
        model_id: str = QWEN_MODEL_ID,
        cache_dir: Path = HUGGINGFACE_CACHE_DIR,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        max_length: int = EMBEDDING_MAX_LENGTH,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model = SentenceTransformer(
            model_name_or_path=self.model_id,
            cache_folder=str(self.cache_dir),
            device=self.device,
            local_files_only=False,
            processor_kwargs={
                "padding_side": "left",
            },
        )

        self.model.max_seq_length = self.max_length

    def _encode(
        self,
        texts: Sequence[str],
        prompt_name: str | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        encode_kwargs = {
            "inputs": list(texts),
            "batch_size": self.batch_size,
            "show_progress_bar": len(texts) > self.batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }

        if prompt_name is not None:
            encode_kwargs["prompt_name"] = prompt_name

        vectors = self.model.encode(
            **encode_kwargs,
        )

        matrix = np.asarray(
            vectors,
            dtype=np.float32,
        )

        return matrix.tolist()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self._encode(
            texts=texts,
        )

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        return self._encode(
            texts=[text],
            prompt_name="query",
        )[0]