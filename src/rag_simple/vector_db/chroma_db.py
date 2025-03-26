from typing import Iterable

import chromadb
from .base import VectorDB
from ..document import Document


class ChromeVectorDB(VectorDB):
    chroma: chromadb.PersistentClient
    embedding_coll: chromadb.Collection

    def connect(self):
        chroma_path = self.embeddings_dir / 'chroma'
        self.chroma = chromadb.PersistentClient(str(chroma_path), database=self.config.db_name)
        self.embedding_coll = self.chroma.get_or_create_collection(
            "chunks",
            metadata={
                "hnsw:space": self.config.hnsw.space,
                "hnsw:construction_ef": self.config.hnsw.construction_ef,
                "hnsw:search_ef": self.config.hnsw.search_ef,
                "hnsw:M": self.config.hnsw.M,
            }
        )

    def clear(self):
        self.chroma.delete_collection("chunks")

    def insert_docs(self, docs: Iterable[Document], embed):
        pass
