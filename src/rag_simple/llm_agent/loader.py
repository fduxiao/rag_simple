from pathlib import Path
from .base import LLMAgent, LLMAgentConfig
from .ollama import OllamaAgent


def get_agent(name, config: LLMAgentConfig) -> LLMAgent:
    if name == "ollama":
        return OllamaAgent(config)
    raise NotImplementedError(f"unknown agent {name}")


class LLMAgentLoader:
    def __init__(self, agents_dir: Path):
        self.agents_dir = agents_dir
        self.loaded_agents: dict[str, LLMAgent] = {}

    def load_agent_by_name(self, name):
        agent = self.loaded_agents.get(name, None)
        if agent is None:
            config = LLMAgentConfig().from_config_file(
                self.agents_dir / f"{name}.toml", write_on_absence=True
            )
            agent = get_agent(name, config)
        return agent

    def connect(self):
        for agent in self.loaded_agents.values():
            agent.connect()

    def close(self):
        for agent in self.loaded_agents.values():
            agent.close()
