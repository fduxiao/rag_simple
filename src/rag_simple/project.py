from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
import tomli_w
import yaml
from .ollama_client import OllamaClient, OllamaConfig
from .embedding import Embedding


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

    def load(self, data: dict):
        self.documents_dir = data.get("documents", self.documents_dir)
        self.embedding_dir = data.get("embeddings", self.embedding_dir)
        self.embedding_model = data.get("embedding_model", self.embedding_model)
        self.embedding_size = data.get("embedding_size", self.embedding_size)
        self.generating_model = data.get("generating_model", self.generating_model)
        self.chromadb_name = data.get("chromadb_name", self.chromadb_name)


class RAGProject:
    Filename = "rag_project.toml"
    OllamaConfigFilename = "ollama.toml"
    Environ = "RAG_PROJECT"

    def __init__(self, project_path: Path | str):
        self.project_path: Path = Path(project_path)
        self.config: RAGProjectConfig = RAGProjectConfig()
        self.ollama_config: OllamaConfig = OllamaConfig()

    @property
    def project_file(self):
        return self.project_path / self.Filename

    @property
    def ollama_config_file(self):
        return self.project_path / self.OllamaConfigFilename

    def write_project_file(self):
        with open(self.project_file, "wb") as file:
            tomli_w.dump(self.config.dump(), file)
        if not self.ollama_config_file.exists():
            with open(self.ollama_config_file, "wb") as file:
                tomli_w.dump(self.ollama_config.dump(), file)

    def load_project_file(self):
        with open(self.project_file, 'rb') as file:
            data = tomllib.load(file)
            self.config.load(data)
        with open(self.ollama_config_file, 'rb') as file:
            data = tomllib.load(file)
            self.ollama_config.load(data)

    @classmethod
    def find_possible_project(cls, path: Path | str = None):
        if path is None:
            path = Path(os.environ.get(cls.Environ, ".")).absolute()
        while True:
            proj = cls(path)
            if proj.project_file.exists():
                proj.load_project_file()
                return proj
            # is the root
            if path.parent == path:
                break
            path = path.parent
        return None

    def parse_dir(self, path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return self.project_path / path

    @property
    def embedding_dir(self) -> Path:
        return self.parse_dir(self.config.embedding_dir)

    @property
    def chroma_dir(self) -> Path:
        return self.embedding_dir / "chroma"

    @property
    def chromedb_name(self) -> str:
        return self.config.chromadb_name

    @property
    def documents_dir(self) -> Path:
        return self.parse_dir(self.config.documents_dir)

    @property
    def project_gitignore(self) -> Path:
        return self.project_path / ".gitignore"

    def init_project(self):
        if self.project_file.exists():
            print(f"Existing project file {self.project_file}.")
            return -1
        self.write_project_file()
        if not self.project_gitignore.exists():
            with open(self.project_gitignore, "w") as file:
                file.write(f"{self.config.embedding_dir}/\n")
                file.write(f"ollama.toml\n")
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
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
            # TODO: write different format with respect to file extension
            yaml.safe_dump_all(data, file)

    def iter_documents(self, base: Path = None):
        if base is None:
            base = self.documents_dir
        one: Path
        for one in base.iterdir():
            if one.is_file():
                if one.suffix in [".yaml", ".yml", ".toml", ".txt"]:
                    yield one
            elif one.is_dir():
                self.iter_documents(one)

    def iter_build_pair(self, run_all):
        for one in self.iter_documents():
            if run_all:
                yield one
                continue
            source_time = os.path.getmtime(one)
            print(source_time)
            yield one

    def build_db(self, dry_run, run_all):
        ollama_client = OllamaClient(self.ollama_config)
        embedding = None
        if not dry_run:
            embedding = Embedding(self.chroma_dir, self.chromedb_name, ollama_client)

        for one in self.iter_build_pair(run_all):
            if dry_run:
                print(one)
                continue
            # build embedding
            embedding.add_document(one)
