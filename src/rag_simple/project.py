import os
from pathlib import Path

import tqdm
import yaml

from .document import DocumentLoader
from .kv_model import KVModel, Field
from .flow_manager import FlowManager
from .llm_agent import LLM, LLMConfig
from .path_builder import PathBuilder
from .vector_db import VectorDBConfig, load_vector_db


class PromptConfig(KVModel):
    retrieval_prefix: str = Field(default="Response based on: ")
    preset: list[dict] = Field(
        default_factory=lambda: [
            {"role": "system", "content": "Response concisely."},
            {
                "role": "system",
                "content": "You will be given some references, related or not related to user input."
                "Response based on those related references.",
            },
        ]
    )


class RAGProjectConfig(KVModel):
    llm: LLMConfig = LLMConfig.as_field()
    vector_db: VectorDBConfig = VectorDBConfig.as_field()
    prompt: PromptConfig = PromptConfig.as_field()


class RAGProject:
    OllamaConfigFilename = "ollama.toml"
    Environ = "RAG_PROJECT"

    def __init__(self, project_path: Path | str, config: RAGProjectConfig = None):
        self.project_path: Path = Path(project_path)
        self.paths = PathBuilder(self.project_path)
        if config is None:
            self.config: RAGProjectConfig = RAGProjectConfig()
            self.load_project_file()
        else:
            self.config: RAGProjectConfig = config
        self.llm = LLM(self.config.llm, self.paths.agents_dir)
        self.vector_db = load_vector_db(
            self.config.vector_db, self.paths.embeddings_dir
        )
        self.flow_manager: FlowManager = FlowManager(
            llm=self.llm, vector_db=self.vector_db
        )

    def write_project_file(self):
        self.config.to_toml(self.paths.project_file)

    def load_project_file(self):
        self.config.from_toml(self.paths.project_file)

    @classmethod
    def find_possible_project(cls, path: Path | str = None):
        if path is None:
            path = Path(os.environ.get(cls.Environ, ".")).absolute()
        while True:
            proj_file_path = path / PathBuilder.ProjectConfigFilename
            if proj_file_path.exists():
                return cls(path)
            # is the root
            if path.parent == path:
                break
            path = path.parent
        return None

    @classmethod
    def init_project(cls, project_path: Path):
        project_path = Path(project_path)
        paths = PathBuilder(project_path)
        if paths.project_file.exists():
            return None
        # prepare files
        paths.init()
        config = RAGProjectConfig()
        config.to_toml(paths.project_file)
        # make the instance
        inst = cls(project_path)
        inst.write_project_file()
        return inst

    @classmethod
    def new(cls, project_path: Path):
        project_path = Path(project_path)
        if project_path.exists():
            return None
        project_path.mkdir(parents=True)
        return cls.init_project(project_path)

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
                        "desc": "put some desired meta data",
                    },
                    "text": "Example\ntext:\nThis will be held by `system`.\n",
                },
                {
                    "metadata": {
                        "role": "system",
                        "desc": "put some desired meta data",
                    },
                    "text": "Another\ndocument.\nNote that YAML uses `--- !tag` to separate documents",
                },
            ]
            # TODO: write different format with respect to file extension
            yaml.safe_dump_all(data, file)

    def build_db(self, dry_run, run_all):
        targets = list(self.paths.iter_build_targets(run_all))
        if len(targets) == 0:
            return
        if dry_run:
            for one in targets:
                print(one)
            return
        loader = DocumentLoader(self.paths.documents_dir)
        # build embedding
        with tqdm.tqdm(targets) as progress:
            for one in targets:
                progress.set_postfix_str(str(one))
                # do things
                rel_path = one.relative_to(self.paths.documents_dir)
                self.flow_manager.remove_by_rel_path(rel_path)
                self.flow_manager.insert_documents(loader.iter_documents(one))
                # set progress
                progress.update()
            progress.set_postfix_str("done")
            progress.refresh()
        self.paths.touch_embeddings_update()

    def retrieve(self, content, limit=5):
        for knowledge in self.flow_manager.retrieve_text(content, limit=limit):
            print(knowledge)

    def clear(self):
        self.flow_manager.clear_db()
        self.paths.embeddings_update_file.unlink(missing_ok=True)

    def ask(self, question, limit):
        chatbot = self.flow_manager.chatbot()
        chatbot.set_retrieval_prefix(self.config.prompt.retrieval_prefix)
        chatbot.extend(self.config.prompt.preset)

        if question is not None:
            for knowledge in chatbot.retrieve(question, limit):
                print(f"{knowledge.metadata['role']}: ", repr(knowledge.text.strip()))
            chatbot.chat(question).print()
            return

        # enter ask-answer loop
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
                user_input = user_input[len("/retrieve ") :]
                chatbot.retrieve(user_input, limit=1)
                continue

            chatbot.retrieve(user_input, limit=limit).drain()

            try:
                chatbot.chat(user_input).print()
            except KeyboardInterrupt:
                continue
