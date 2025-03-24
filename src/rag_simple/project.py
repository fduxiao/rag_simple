from dataclasses import dataclass
from pathlib import Path
import tomllib
import tomli_w
import yaml


@dataclass
class RAGProjectConfig:
    documents_dir: str = "documents"
    embedding_dir: str = "embeddings"
    embedding_model: str = "mxbai-embed-large"
    embedding_size: int = 1024
    generating_model: str = "deepseek-r1:7b"
    chromadb_name: str = "default_database"

    def dump(self):
        return {
            "documents": self.documents_dir,
            "embeddings": self.embedding_dir,
            "embedding_model": self.embedding_model,
            "embedding_size": self.embedding_size,
            "generating_model": self.generating_model,
            "chromadb_name": self.chromadb_name,
        }


class RAGProject:
    def __init__(self, project_path: Path | str):
        self.project_path: Path = Path(project_path)
        self.config: RAGProjectConfig = RAGProjectConfig()

    @property
    def project_file(self):
        return self.project_path / "rag_project.toml"

    def write_project_file(self):
        with open(self.project_file, "wb") as file:
            tomli_w.dump(self.config.dump(), file)

    def parse_dir(self, path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return self.project_path / path

    @property
    def embedding_dir(self) -> Path:
        return self.parse_dir(self.config.embedding_dir)

    @property
    def documents_dir(self) -> Path:
        return self.parse_dir(self.config.documents_dir)

    def init_project(self):
        if self.project_file.exists():
            print(f"Existing project file {self.project_file}.")
            return -1
        self.write_project_file()
        with open(self.project_path / ".gitignore", "w") as file:
            file.write(f"{self.config.embedding_dir}/\n")
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    def new_project(self):
        if self.project_path.exists():
            print(f"Existing project file {self.project_file}.")
            return -1
        self.project_path.mkdir(parents=True)
        self.init_project()

    @staticmethod
    def new_doc(path: Path | str, force=False):
        path = Path(path)
        if path.suffix not in {".toml", ".yaml", ".yml"}:
            path = Path(str(path) + ".yaml")

        if not path.parent.exists():
            print(f"Parent directory does not exist {path}.")
            return -1

        if path.exists() and not force:
            print(f"Existing document file {path}.")
            return -1

        with open(path, "w") as file:
            data = [
                {
                "metadata": {
                    "role": "system",
                    "desc": "put some desired meta data"
                },
                "text": "Example\ntext:\nThis will be held by `system`.\n",
                },
                {
                    "metadata": {
                        "role": "system",
                        "desc": "put some desired meta data"
                    },
                    "text": "Another\ndocument.\nNote that YAML uses `---` to separate documents",
                },
            ]

            yaml.safe_dump_all(data, file)
