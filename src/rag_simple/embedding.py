from pathlib import Path
import chromadb
from .ollama_client import OllamaClient


class Embedding:
    def __init__(self, chroma_path: Path, chrome_db: str, ollama_client: OllamaClient):
        self.chroma = chromadb.PersistentClient(str(chroma_path), database=chrome_db)
        self.embedding = self.chroma.get_or_create_collection("embedding")
        self.ollama_client = ollama_client

    def add_document(self, doc_path: Path):
        pass
