from pathlib import Path

from .base import BaseVectorDB, VectorDB, VectorDBConfig
from .chroma_db import ChromeVectorDB


def load_vector_db(config: VectorDBConfig, embeddings_dir: Path):
    if config.engine == "chroma":
        return ChromeVectorDB(config, embeddings_dir)
    raise NotImplemented(f"unknown vector database {config.engine}")
