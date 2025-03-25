import ollama
from .base import LLMAgent, LLMAgentConfig


class OllamaAgent(LLMAgent):
    def __init__(self, config: LLMAgentConfig):
        super().__init__(config)
        self.client = ollama.Client(
            host=config.api_url,
            headers=config.headers
        )

    def embed(self, model, texts: list[str]) -> list[list[float]]:
        resp = self.client.embed(model=model, input=texts)
        embeddings = resp["embeddings"]
        return embeddings

    def chat(self, model, messages):
        stream = self.client.chat(
            model=model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            yield chunk
