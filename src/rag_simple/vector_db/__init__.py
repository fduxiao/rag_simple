from pathlib import Path

from .base import BaseVectorDB, VectorDB, VectorDBConfig
from .chroma_db import ChromeVectorDB


__all__ = [
    "BaseVectorDB", "VectorDB", "VectorDBConfig",
    "ChromeVectorDB",
    "load_vector_db"
]


def load_vector_db(config: VectorDBConfig, embeddings_dir: Path):
    if config.engine == "chroma":
        return ChromeVectorDB(config, embeddings_dir)
    raise NotImplementedError(f"unknown vector database {config.engine}")
