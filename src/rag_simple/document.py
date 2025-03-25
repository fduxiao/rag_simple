from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class DocumentSentence:
    rel_path: str
    doc_id: str
    sentence_index: int
    text: str

    @property
    def id(self):
        return f"{self.doc_id}|{self.sentence_index}"

    def dump(self):
        return {
            "rel_path": self.rel_path,
            "doc_id": self.doc_id,
            "sentence_index": self.sentence_index,
        }


@dataclass
class Document:
    rel_path: str
    index: int
    text: str
    metadata: dict

    @property
    def doc_id(self):
        return f"{self.rel_path}|{self.index}"

    @property
    def id(self):
        return f"{self.doc_id}|0"

    def __post_init__(self):
        self.metadata["doc_id"] = self.doc_id
        self.metadata["rel_path"] = str(self.rel_path)
        self.metadata["doc_index"] = self.index
        self.metadata["sentence_index"] = 0

    def iter_doc_sentences(self):
        lines = self.text.strip().split("\n")
        if len(lines) == 1:
            return
        for i, sentence in enumerate(lines):
            yield DocumentSentence(
                self.rel_path,
                self.doc_id,
                i + 1,
                sentence,
            )


class DocumentLoader:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    @staticmethod
    def load_obj_stream_from_file(path: Path):
        # TODO: read by file extension
        with open(path, "r") as file:
            for index, one in enumerate(yaml.safe_load_all(file)):
                yield one

    def iter_documents(self, path: Path):
        rel_path = path.relative_to(self.base_dir)
        for index, one in enumerate(self.load_obj_stream_from_file(path)):
            text = one["text"]
            metadata: dict = one.get("metadata", {})
            yield Document(str(rel_path), index, text, metadata)
