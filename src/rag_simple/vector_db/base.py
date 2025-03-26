from typing import Iterable
from pathlib import Path

from ..document import Document
from ..kv_model import KVModel, Field


class HNSWConfig(KVModel):
    space: str = Field(default="l2")
    construction_ef: int = Field(default=100)
    search_ef: int = Field(default=100)
    M: int = Field(default=16)


class VectorDBConfig(KVModel):
    engine: str = Field(default="chroma")
    db_name: str = Field(default="default_database")
    hnsw: HNSWConfig = HNSWConfig.as_field()


class BaseVectorDB:
    def connect(self):
        pass

    def close(self):
        pass

    def clear(self):
        pass

    def insert_docs(self, docs: Iterable[Document], embed):
        pass

    def find_one(self):
        pass


class VectorDB(BaseVectorDB):
    def __init__(self, config: VectorDBConfig, embeddings_dir: Path):
        self.embeddings_dir: Path = embeddings_dir
        self.config: VectorDBConfig = config
