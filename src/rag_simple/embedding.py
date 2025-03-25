from pathlib import Path
import chromadb

from .document import DocumentLoader, Document
from .prompt import Knowledge
from .kv_model import KVModel, Field


class HNSWConfig(KVModel):
    space: str = Field(default="l2")
    construction_ef: int = Field(default=100)
    search_ef: int = Field(default=100)
    M: int = Field(default=16)


class ChromaConfig(KVModel):
    db_name: str = Field(default="default_database")
    hnsw: HNSWConfig = HNSWConfig.as_field()


class EmbeddingDB:
    def __init__(self, documents_dir: Path, chroma_path: Path, chrome_config: ChromaConfig):
        self.documents_dir = documents_dir
        self.chroma = chromadb.PersistentClient(str(chroma_path), database=chrome_config.db_name)
        self.embedding_coll = self.chroma.get_or_create_collection(
            "chunks",
            metadata={
                "hnsw:space": chrome_config.hnsw.space,
                "hnsw:construction_ef": chrome_config.hnsw.construction_ef,
                "hnsw:search_ef": chrome_config.hnsw.search_ef,
                "hnsw:M": chrome_config.hnsw.M,
            }
        )

    def clear(self):
        self.chroma.delete_collection("chunks")

    def add_document(self, doc: Document, embed):
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

    def add_doc_file(self, doc_path: Path, embed):
        # clear old data
        rel_path = doc_path.relative_to(self.documents_dir)
        self.embedding_coll.delete(where={"rel_path": str(rel_path)})

        loader = DocumentLoader(self.documents_dir)
        for doc in loader.iter_documents(doc_path):
            self.add_document(doc, embed)

    def retrieve_one(self, embedding, escaping=None):
        if escaping is None:
            escaping = []
        where = None
        if len(escaping) != 0:
            where = {"doc_id": {"$nin": escaping}}
        results = self.embedding_coll.query(
            query_embeddings=embedding,
            n_results=1,
            where=where
        )
        metadata = results["metadatas"][0]
        if len(metadata) == 0:
            return None
        metadata = metadata[0]
        data_id = results["ids"][0][0]
        text = results["documents"][0][0]
        dist = results["distances"][0][0]
        doc_id = metadata["doc_id"]
        if metadata["sentence_index"] != 0:
            results = self.embedding_coll.get(
                ids=[f'{doc_id}|0'],
            )
            data_id = results["ids"][0]
            text = results["documents"][0]
            metadata = results["metadatas"][0]
        return doc_id, data_id, text, metadata, dist

    def retrieve_by_sentence(self, embedding, limit=5, escaping=None):
        if escaping is None:
            escaping = []
        for i in range(limit):
            one = self.retrieve_one(embedding, escaping)
            if one is None:
                continue
            doc_id, data_id, text, metadata, dist = one
            yield Knowledge(doc_id, text, metadata, dist)
            escaping.append(doc_id)

    def retrieve_doc(self, embedding, limit=5):
        results = self.embedding_coll.query(
            query_embeddings=embedding,
            n_results=limit,
            where={"sentence_index": 0}
        )
        escaping = []
        for data_id, text, metadata, dist in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            doc_id = metadata["doc_id"]
            yield Knowledge(doc_id, text, metadata, dist)
            escaping.append(doc_id)
        return escaping

    def retrieve(self, embedding, limit=5):
        escaping = yield from self.retrieve_doc(embedding, limit)
        yield from self.retrieve_by_sentence(embedding, limit, escaping)
