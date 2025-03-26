from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class PathBuilder:
    ProjectConfigFilename = "rag_project.toml"

    project_path: Path

    @property
    def project_file(self):
        return self.project_path / self.ProjectConfigFilename

    @property
    def project_gitignore(self) -> Path:
        return self.project_path / ".gitignore"

    def parse_dir(self, path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return self.project_path / path

    @property
    def embeddings_dir(self) -> Path:
        return self.project_path / "embeddings"

    @property
    def documents_dir(self) -> Path:
        return self.project_path / "documents"

    @property
    def embeddings_update_file(self) -> Path:
        return self.embeddings_dir / "update.time"

    @property
    def agents_dir(self):
        return self.project_path / 'agents'

    @property
    def agent_gitignore(self) -> Path:
        return self.agents_dir / ".gitignore"

    def init(self):
        if not self.project_gitignore.exists():
            with open(self.project_gitignore, "w") as file:
                file.write(f"{self.embeddings_dir}/\n")
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        if not self.agent_gitignore.exists():
            with open(self.agent_gitignore, "w") as file:
                file.write(f"ollama.toml\n")

    # iterate all documents
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

    def touch_embeddings_update(self):
        self.embeddings_update_file.touch()

    # find outdated documents
    def iter_build_targets(self, run_all):
        target_time = None
        if self.embeddings_update_file.exists():
            target_time = os.path.getmtime(self.embeddings_update_file)
        for one in self.iter_documents():
            if run_all:
                yield one
                continue
            source_time = os.path.getmtime(one)
            if target_time is None or target_time < source_time:
                yield one
