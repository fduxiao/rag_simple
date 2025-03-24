from dataclasses import dataclass
import datetime
from pathlib import Path
from typing import Any, Mapping
import chromadb
import yaml

from .ollama_client import OllamaClient


@dataclass
class Knowledge:
    id: str
    text: str
    metadata: Mapping[str, Any]
    dist: float


class EmbeddingDB:
    def __init__(self, documents_dir: Path, chroma_path: Path, chrome_db: str):
        self.documents_dir = documents_dir
        self.chroma = chromadb.PersistentClient(str(chroma_path), database=chrome_db)
        self.embedding_coll = self.chroma.get_or_create_collection("chunks")

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
