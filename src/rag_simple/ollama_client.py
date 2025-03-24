import ollama


class OllamaClient:
    def __init__(self):
        self.client = ollama.Client()
