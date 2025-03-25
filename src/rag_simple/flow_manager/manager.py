from pathlib import Path
from typing import Optional

from ..llm_agent import LLMAgent, LLMAgentLoader
from ..kv_model import KVModel, Field


class EmbedConfig(KVModel):
    agent: str = Field(default="ollama")
    model: str = Field(default="mxbai-embed-large")
    size: int = Field(default=1024)


class ChatConfig(KVModel):
    agent: str = Field(default="ollama")
    model: str = Field(default="deepseek-r1:7b")


class FlowConfig(KVModel):
    embed: EmbedConfig = EmbedConfig.as_field()
    chat: ChatConfig = ChatConfig.as_field()


class FlowManager:
    def __init__(self, agent_loader_class=LLMAgentLoader):
        self.config: Optional[FlowConfig] = None
        self.agent_loader_class = agent_loader_class
        self.agent_loader: Optional[LLMAgentLoader] = None
        self.embedding_agent: Optional[LLMAgent] = None
        self.chatting_agent: Optional[LLMAgent] = None
        self.is_setup = False

    def set_config(self, config: FlowConfig, agents_dir: Path):
        self.config = config
        self.agent_loader = self.agent_loader_class(agents_dir)

    def load_agents(self):
        self.embedding_agent = self.agent_loader.load_agent_by_name(self.config.embed.agent)
        self.chatting_agent = self.agent_loader.load_agent_by_name(self.config.chat.agent)

    def connect(self):
        self.agent_loader.connect()

    def setup(self):
        if self.is_setup:
            return
        self.load_agents()
        self.connect()
        self.is_setup = True

    def close(self):
        if not self.is_setup:
            return
        self.agent_loader.close()

    def __del__(self):
        self.close()

    def embed(self, input_text: list[str]) -> list[list[float]]:
        self.setup()
        return self.embedding_agent.embed(self.config.embed.model, input_text)

    def chat(self, messages):
        self.setup()
        return self.chatting_agent.chat(self.config.chat.model, messages)
