from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class DocumentSentence:
    doc_id: str
    sentence_index: int
    text: str

    @property
    def id(self):
        return f"{self.doc_id}|{self.sentence_index}"

    def dump(self):
        return {
            "doc_id": self.doc_id,
            "sentence_index": self.sentence_index,
        }


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict

    @property
    def id(self):
        return f"{self.doc_id}|0"

    def iter_doc_sentences(self):
        lines = self.text.strip().split("\n")
        if len(lines) == 1:
            return
        for i, sentence in enumerate(lines):
            yield DocumentSentence(
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
            doc_id =  f"{rel_path}|{index}"
            metadata["doc_id"] = doc_id
            metadata["rel_path"] = str(rel_path)
            metadata["doc_index"] = index
            metadata["sentence_index"] = 0
            yield Document(doc_id, text, metadata)
