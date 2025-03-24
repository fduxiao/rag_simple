import os
from pathlib import Path
import tomllib
import tomli_w
import tqdm
import yaml

from .kv_model import KVModel, Field
from .ollama_client import OllamaClient, OllamaConfig
from .embedding import EmbeddingDB, ChromaConfig
from .prompt import Prompt


class RAGProjectConfig(KVModel):
    documents_dir: str = Field(default="documents")
    embeddings_dir: str = Field(default="embeddings")
    embedding_model: str = Field(default="mxbai-embed-large")
    embedding_size: int = Field(default=1024)
    generating_model: str = Field(default="deepseek-r1:7b")
    chromadb_config: ChromaConfig = ChromaConfig.as_field()


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
    def embeddings_dir(self) -> Path:
        return self.parse_dir(self.config.embeddings_dir)

    @property
    def chroma_dir(self) -> Path:
        return self.embeddings_dir / "chroma"

    @property
    def chroma_config(self) -> ChromaConfig:
        return self.config.chromadb_config

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
                file.write(f"{self.config.embeddings_dir}/\n")
                file.write(f"ollama.toml\n")
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
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
                    "text": "Another\ndocument.\nNote that YAML uses `--- !tag` to separate documents",
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

    def iter_build_targets(self, run_all, embedding_db: EmbeddingDB):
        for one in self.iter_documents():
            if run_all:
                yield one
                continue
            source_time = os.path.getmtime(one)
            target_time = embedding_db.document_mtime(one)
            if target_time is None or target_time < source_time:
                yield one

    @property
    def embedding_model(self):
        return self.config.embedding_model

    @property
    def generating_model(self):
        return self.config.generating_model

    def build_db(self, dry_run, run_all):
        ollama_client = OllamaClient(self.ollama_config)
        embedding_db = EmbeddingDB(self.documents_dir, self.chroma_dir, self.chroma_config)

        targets = list(self.iter_build_targets(run_all, embedding_db))
        if dry_run:
            for one in targets:
                print(one)
            return
        # build embedding
        with tqdm.tqdm(targets) as progress:
            for one in targets:
                progress.set_postfix_str(str(one))
                embedding_db.add_document(one, ollama_client, self.embedding_model)
                progress.update()
            progress.set_postfix_str("done")
            progress.refresh()

    def retrieve(self, content, limit=5):
        ollama_client = OllamaClient(self.ollama_config)
        embedding_db = EmbeddingDB(self.documents_dir, self.chroma_dir, self.chroma_config)

        embedding = ollama_client.embed(self.embedding_model, content)
        for knowledge in embedding_db.retrieve(embedding, limit=limit):
            print(knowledge)

    def clear(self):
        embedding_db = EmbeddingDB(self.documents_dir, self.chroma_dir, self.chroma_config)
        embedding_db.clear()

    def ask(self, question):
        ollama_client = OllamaClient(self.ollama_config)
        embedding_db = EmbeddingDB(self.documents_dir, self.chroma_dir, self.chroma_config)

        if question is not None:
            # make embedding
            embedding = ollama_client.embed(self.embedding_model, question)
            prompt = Prompt()
            for knowledge in embedding_db.retrieve(embedding, limit=5):
                prompt.add_knowledge(knowledge)
            for chunk in ollama_client.chat(self.generating_model, prompt):
                print(chunk['message']['content'], end='', flush=True)
            return

        # enter ask-answer loop
        prompt = Prompt()
        retrieved = set()
        while True:
            try:
                user_input = input(">>> ")
            except KeyboardInterrupt:
                print()
                continue
            except EOFError:
                print()
                return

            # parse user input
            if user_input.startswith("/retrieve "):
                user_input = user_input[len('/retrieve '):]
                # add knowledge
                embedding = ollama_client.embed(self.embedding_model, user_input)
                for knowledge in embedding_db.retrieve(embedding, limit=1):
                    if knowledge.id not in retrieved:
                        prompt.add_knowledge(knowledge)
                        retrieved.add(knowledge.id)
                continue

            prompt.add_message(user_input, role="user")

            try:
                response = ""
                for chunk in ollama_client.chat(self.generating_model, prompt):
                    content = chunk['message']['content']
                    response += content
                    print(content, end='', flush=True)
                print()
            except KeyboardInterrupt:
                continue

            prompt.add_message(response, role="assistant")
