from functools import wraps
from io import IOBase
import sys
from typing import Any, Iterable, Callable, Protocol
from .prompt import Knowledge, Prompt


ChatFunc = Callable[[Prompt], Iterable[str]]


class RetrieveFunc(Protocol):
    def __call__(self, text, *, limit=5) -> Iterable[Knowledge]:
        pass


class Stream:
    def __init__(self, stream: Iterable[Any] = None):
        self.stream = stream

    def __iter__(self):
        return iter(self.stream)

    def drain(self):
        iterator = iter(self.stream)
        while True:
            try:
                next(iterator)
            except StopIteration as err:
                return err.value


def make_stream(func):
    """
    A decorator turning a generator to a stream object

    :param func:
    :return:
    """

    @wraps(func)
    def wrapped(*args, **kwargs) -> Stream:
        return Stream(func(*args, **kwargs))

    return wrapped


class Response(Stream):
    def __init__(self, chatbot: "Chatbot", stream: Iterable = None):
        super().__init__(stream)
        self.chatbot = chatbot

    def __iter__(self):
        return self.stream

    def iter_message(self):
        total = ""
        for content in self.stream:
            yield content
            total += content
        self.chatbot.add_assistant_message(total)

    def print(self, file: IOBase = None, end="\n"):
        if self.stream is None:
            return
        if file is None:
            file = sys.stdout
        for content in self.iter_message():
            print(content, end="", flush=True, file=file)
        print(end, end="", file=file)


class Chatbot:
    def __init__(self, chat: ChatFunc, retrieve: RetrieveFunc):
        self.chat_func = chat
        self.retrieve_func = retrieve
        self.messages = Prompt()
        self.added_knowledge = set()
        self.retrieval_prefix = ""

    def set_retrieval_prefix(self, prefix):
        self.retrieval_prefix = prefix
        return self

    def extend(self, iterable):
        self.messages.extend(iterable)
        return self

    def add_assistant_message(self, text):
        self.messages.add_message(text, role="assistant")

    @make_stream
    def retrieve(self, text, limit=5) -> Stream:
        for knowledge in self.retrieve_func(text, limit=limit):
            yield knowledge
            if knowledge.id not in self.added_knowledge:
                knowledge.set_prefix(self.retrieval_prefix)
                self.messages.add_knowledge(knowledge)
                self.added_knowledge.add(knowledge.id)

    def chat(self, text) -> Response:
        self.messages.add_message(text, role="user")
        stream = self.chat_func(self.messages)
        return Response(self, stream)
