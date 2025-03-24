from dataclasses import dataclass, field
import ollama


@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    headers: dict = field(default_factory=dict)

    def dump(self):
        return {
            "host": self.host,
            "headers": self.headers
        }

    def load(self, data: dict):
        self.host = data.get("host", self.host)
        self.headers = data.get("headers", self.headers)


class OllamaClient:
    def __init__(self, config: OllamaConfig):
        self.client = ollama.Client(
            host=config.host,
            headers=config.headers
        )
