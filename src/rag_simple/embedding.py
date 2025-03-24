from dataclasses import dataclass
import datetime
from pathlib import Path
import chromadb
import yaml

from .ollama_client import OllamaClient
from.prompt import Knowledge


@dataclass
class ChromaConfig:
    db_name: str = "default_database"
    space: str = "l2"  # l2, ip, cosine
    construction_ef: int = 100
    search_ef: int = 100
    M: int = 16

    def dump(self):
        return {
            "name": self.db_name,
            "hnsw": {
                "space": self.space,
                "construction_ef": self.construction_ef,
                "search_ef": self.search_ef,
                "M": self.M,
            }
        }

    def load(self, data: dict):
        self.db_name = data.get("name", self.db_name)
        hnsw = data.get("hnsw", {})
        self.space = hnsw.get("space", self.space)
        self.construction_ef = hnsw.get("construction_ef", self.construction_ef)
        self.search_ef = hnsw.get("search_ef", self.search_ef)
        self.M = hnsw.get("M", self.M)


class EmbeddingDB:
    def __init__(self, documents_dir: Path, chroma_path: Path, chrome_config: ChromaConfig):
        self.documents_dir = documents_dir
        self.chroma = chromadb.PersistentClient(str(chroma_path), database=chrome_config.db_name)
        self.embedding_coll = self.chroma.get_or_create_collection(
            "chunks",
            metadata={
                "hnsw:space": chrome_config.space,
                "hnsw:construction_ef": chrome_config.construction_ef,
                "hnsw:search_ef": chrome_config.search_ef,
                "hnsw:M": chrome_config.M,
            }
        )

    def clear(self):
        self.chroma.delete_collection("chunks")

    def document_mtime(self, doc_path: Path):
        rel_path = doc_path.relative_to(self.documents_dir)
        data = self.embedding_coll.get(where={"rel_path": str(rel_path)})
        metadatas = data["metadatas"]
        mtime = [x.get("mtime", 0) for x in metadatas]
        mtime.sort()
        if len(mtime) == 0:
            return None
        return mtime[0]  # smallest

    def read_document_chunk(self, doc_path: Path):
        rel_path = doc_path.relative_to(self.documents_dir)
        mtime = datetime.datetime.now(datetime.UTC).timestamp()
        # TODO: read by file extension
        with open(doc_path, "r") as file:
            for index, one in enumerate(yaml.safe_load_all(file)):
                data_id = f'{rel_path}_{index}'
                text = one["text"]
                metadata = one.get("metadata", {})
                metadata.update({
                    "index": index,
                    "rel_path": str(rel_path),
                    "mtime": mtime
                })
                yield data_id, text, metadata

    def add_document(self, doc_path: Path, ollama_client: OllamaClient, model):
        rel_path = doc_path.relative_to(self.documents_dir)
        # first clear old data
        self.embedding_coll.delete(where={"rel_path": str(rel_path)})
        for chunk in self.read_document_chunk(doc_path):
            data_id, text, metadata = chunk
            embedding = ollama_client.embed(model, text)
            self.embedding_coll.add(
                ids=[data_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text],
            )

    def retrieve(self, embedding, limit=5):
        results = self.embedding_coll.query(
            query_embeddings=[embedding],
            n_results=limit
        )
        for data_id, text, metadata, dist in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            yield Knowledge(data_id, text, metadata, dist)
