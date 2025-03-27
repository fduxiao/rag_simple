from ..chatbot import Chatbot
from .respond import ChatResponder


class Repl:
    def __init__(self, chatbot: Chatbot, default_limit=3):
        self.responder = ChatResponder(chatbot, default_limit)

    @staticmethod
    def read_input():
        try:
            return input(">>> ")
        except EOFError:
            return None

    def read_valid_input(self):
        while True:
            try:
                user_input = self.read_input()
                if user_input is None:
                    return None
                if user_input == "":
                    continue
                return user_input
            except KeyboardInterrupt:
                print()
                continue

    def loop(self):
        while True:
            user_input = self.read_valid_input()
            if user_input is None:
                break

            try:
                self.responder.respond_to(user_input)
            except KeyboardInterrupt:
                print()
