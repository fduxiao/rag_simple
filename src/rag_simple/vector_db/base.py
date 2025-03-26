from dataclasses import dataclass
from typing import List, Iterable, Any, Mapping
from pathlib import Path

from ..document import Document
from ..kv_model import KVModel, Field
from ..prompt import Knowledge


@dataclass
class QueryResult:
    ids: List[List[str]]
    embeddings: List[List[Any]]
    texts: List[List[str]]
    metadatas: List[List[Mapping[str, Any]]] = None
    distances: List[List[float]] = None


@dataclass
class FindResult:
    ids: List[str]
    embeddings: List[Any]
    texts: List[str]
    metadatas: List[Mapping[str, Any]] = None


class BaseVectorDB:
    def connect(self):
        pass

    def close(self):
        pass

    def clear(self):
        pass

    def remove_by_rel_path(self, rel_path: str | Path):
        pass

    def insert_documents(self, docs: Iterable[Document], embed):
        pass

    def query_embeddings(self, embeddings, where, n_results) -> QueryResult:
        pass

    def find_by_ids(self, ids) -> FindResult:
        pass

    def retrieve(self, embedding, *, limit=5) -> Iterable[Knowledge]:
        pass


class VectorDBSearch(BaseVectorDB):
    def retrieve_one(self, embedding, escaping=None):
        if escaping is None:
            escaping = []
        where = None
        if len(escaping) != 0:
            where = {"doc_id": {"$nin": escaping}}
        results = self.query_embeddings(
            embeddings=embedding,
            n_results=1,
            where=where
        )
        metadata = results.metadatas[0]
        if len(metadata) == 0:
            return None
        metadata = metadata[0]
        data_id = results.ids[0][0]
        text = results.texts[0][0]
        dist = results.distances[0][0]
        doc_id = metadata["doc_id"]
        if metadata["sentence_index"] != 0:
            results = self.find_by_ids([f'{doc_id}|0'])
            data_id = results.ids[0]
            text = results.texts[0]
            metadata = results.metadatas[0]
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
        results = self.query_embeddings(
            embeddings=embedding,
            n_results=limit,
            where={"sentence_index": 0}
        )
        escaping = []
        for data_id, text, metadata, dist in zip(
            results.ids[0], results.texts[0], results.metadatas[0], results.distances[0]
        ):
            doc_id = metadata["doc_id"]
            yield Knowledge(doc_id, text, metadata, dist)
            escaping.append(doc_id)
        return escaping

    def retrieve(self, embedding, *, limit=5) -> Iterable[Knowledge]:
        escaping = yield from self.retrieve_doc(embedding, limit)
        yield from self.retrieve_by_sentence(embedding, limit, escaping)


class HNSWConfig(KVModel):
    space: str = Field(default="l2")
    construction_ef: int = Field(default=100)
    search_ef: int = Field(default=100)
    M: int = Field(default=16)


class VectorDBConfig(KVModel):
    engine: str = Field(default="chroma")
    db_name: str = Field(default="default_database")
    hnsw: HNSWConfig = HNSWConfig.as_field()


class VectorDB(VectorDBSearch):
    def __init__(self, config: VectorDBConfig, embeddings_dir: Path):
        self.embeddings_dir: Path = embeddings_dir
        self.config: VectorDBConfig = config
