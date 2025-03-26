from pathlib import Path
from typing import Iterable


import chromadb
from .base import VectorDB, QueryResult, FindResult
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

    def remove_by_rel_path(self, rel_path: str | Path):
        self.embedding_coll.delete(where={"rel_path": str(rel_path)})

    def insert_documents(self, docs: Iterable[Document], embed):
        for doc in docs:
            embedding = embed([doc.text])
            self.embedding_coll.add(
                ids=[doc.id],
                embeddings=embedding,
                metadatas=[doc.metadata],
                documents=[doc.text]
            )
            for sentence in doc.iter_doc_sentences():
                embedding = embed([sentence.text])
                self.embedding_coll.add(
                    ids=[sentence.id],
                    embeddings=embedding,
                    metadatas=[sentence.dump()],
                    documents=[sentence.text]
                )

    def query_embeddings(self, embeddings, where, n_results) -> QueryResult:
        result = self.embedding_coll.query(
            query_embeddings=embeddings,
            n_results=n_results,
            where=where
        )
        return QueryResult(
            result["ids"],
            result["embeddings"],
            result["documents"],
            result["metadatas"],
            result["distances"],
        )

    def find_by_ids(self, ids) -> FindResult:
        result = self.embedding_coll.get(
            ids=ids,
        )
        return FindResult(
            result["ids"],
            result["embeddings"],
            result["documents"],
            result["metadatas"],
        )
