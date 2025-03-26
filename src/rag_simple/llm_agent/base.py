from typing import Iterable

from ..kv_model import KVModel, Field
from ..prompt import Prompt


class LLMAgentConfig(KVModel, Field):
    api_url: str = Field(default="http://localhost:11434")
    model_dir: str = Field(default="")
    headers: str = Field(default_factory=lambda: {"X-Some-Header": "some_secret"})


class LLMAgent:
    def __init__(self, config: LLMAgentConfig):
        self.config = config

    def connect(self):
        pass

    def close(self):
        pass

    def embed(self, model, texts: list[str]) -> list[list[float]]:
        pass

    def chat(self, model, messages: Prompt) -> Iterable[str]:
        pass
