from typing import Any, Mapping
from dataclasses import dataclass


@dataclass
class Knowledge:
    id: str
    text: str
    metadata: Mapping[str, Any]
    dist: float

    def to_prompt(self):
        return {
            "role": self.metadata.get("role", "system"),
            "content": self.text
        }


class Prompt:
    def __init__(self):
        self.messages = []

    def add_message(self, content, role="user"):
        self.messages.append({
            "role": role,
            "content": content
        })
        return self

    def add_knowledge(self, knowledge: Knowledge):
        self.messages.append(knowledge.to_prompt())
        return self

    def __iter__(self):
        return iter(self.messages)
