from typing import Iterable
from pathlib import Path

from ..document import Document
from ..llm_agent import BaseLLM
from ..prompt import Knowledge
from ..vector_db import BaseVectorDB


class FlowManager:
    def __init__(self, llm: BaseLLM, vector_db: BaseVectorDB):
        self.llm = llm
        self.vector_db = vector_db
        self.is_setup = False

    def connect(self):
        self.llm.connect()
        self.vector_db.connect()

    def setup(self):
        if self.is_setup:
            return
        self.connect()
        self.is_setup = True

    def close(self):
        if not self.is_setup:
            return
        self.llm.close()
        self.vector_db.close()

    def __del__(self):
        self.close()

    def embed(self, input_text: list[str]) -> list[list[float]]:
        self.setup()
        return self.llm.embed(input_text)

    def chat(self, messages):
        self.setup()
        return self.llm.chat(messages)

    def clear_db(self):
        self.setup()
        self.vector_db.clear()

    def remove_by_rel_path(self, rel_path: str | Path):
        self.setup()
        self.vector_db.remove_by_rel_path(rel_path)

    def insert_documents(self, docs: Iterable[Document]):
        self.setup()
        return self.vector_db.insert_documents(docs, self.embed)

    def retrieve_text(self, text, limit=5) -> Iterable[Knowledge]:
        self.setup()
        embedding = self.embed([text])
        return self.vector_db.retrieve(embedding, limit=limit)
