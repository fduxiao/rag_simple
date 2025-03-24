import ollama
from .kv_model import KVModel, Field


class OllamaConfig(KVModel):
    host: str = Field(default="http://localhost:11434")
    headers: dict = Field(default_factory=dict)


class OllamaClient:
    def __init__(self, config: OllamaConfig):
        self.client = ollama.Client(
            host=config.host,
            headers=config.headers
        )

    def embed(self, model, text):
        resp = self.client.embed(model=model, input=text)
        embeddings = resp["embeddings"][0]
        return embeddings

    def chat(self, model, messages):
        stream = self.client.chat(
            model=model,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            yield chunk
