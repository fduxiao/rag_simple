from pathlib import Path
from typing import Iterable

from ..kv_model import KVModel, Field
from .loader import LLMAgentLoader
from ..prompt import Prompt


class EmbedConfig(KVModel):
    agent: str = Field(default="ollama")
    model: str = Field(default="mxbai-embed-large")
    size: int = Field(default=1024)


class ChatConfig(KVModel):
    agent: str = Field(default="ollama")
    model: str = Field(default="deepseek-r1:7b")


class LLMConfig(KVModel):
    embed: EmbedConfig = EmbedConfig.as_field()
    chat: ChatConfig = ChatConfig.as_field()


class BaseLLM:
    def connect(self):
        pass

    def close(self):
        pass

    def embed(self, input_text: list[str]) -> list[list[float]]:
        pass

    def chat(self, messages: Prompt) -> Iterable[str]:
        pass


class LLM(BaseLLM):
    def __init__(self, config: LLMConfig, agents_dir: Path, loader_class=LLMAgentLoader):
        self.config = config
        self.agents_dir = agents_dir
        self.agent_loader = loader_class(agents_dir)

        self.embedding_agent = self.agent_loader.load_agent_by_name(self.config.embed.agent)
        self.chatting_agent = self.agent_loader.load_agent_by_name(self.config.chat.agent)

    def connect(self):
        self.agent_loader.connect()

    def close(self):
        self.agent_loader.close()

    def embed(self, input_text: list[str]) -> list[list[float]]:
        return self.embedding_agent.embed(self.config.embed.model, input_text)

    def chat(self, messages):
        return self.chatting_agent.chat(self.config.chat.model, messages)
